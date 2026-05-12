use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

use bytemuck::Zeroable;

use crate::sdf::TileKey;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: [u8; 4] = *b"SDFC";
const VERSION: u32 = 1;

/// Max in-flight requests to avoid memory spikes.
const MAX_INFLIGHT: usize = 32;

// ---------------------------------------------------------------------------
// Cache file header (on-disk)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CacheHeader {
    magic: [u8; 4],
    version: u32,
    world_origin_x: f32,
    world_origin_y: f32,
    world_size_x: f32,
    world_size_y: f32,
    tile_size: f32,
    sdf_resolution: u32,
    road_id_resolution: u32,
    grid_w: u32,
    grid_h: u32,
    _pad: u32,
}

/// One entry in the on-disk index table.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct IndexEntry {
    /// Byte offset from start of file to the compressed blob. 0 = not present.
    offset: u64,
    /// Compressed size in bytes. 0 = not present.
    compressed_size: u32,
    _pad: u32,
}

// ---------------------------------------------------------------------------
// Tile data returned from cache reads
// ---------------------------------------------------------------------------

/// Decompressed tile pixel data ready for GPU upload.
pub struct TileData {
    /// SDF pixels: `sdf_resolution² × 4` bytes (RG16F = 2 channels × 2 bytes).
    pub sdf_pixels: Vec<u8>,
    /// Road ID pixels: `road_id_resolution² × 2` bytes (R16_SFLOAT = 1 channel × 2 bytes).
    pub road_id_pixels: Vec<u8>,
}

// ---------------------------------------------------------------------------
// SdfCacheFile — indexed binary cache with zstd compression
// ---------------------------------------------------------------------------

pub struct SdfCacheFile {
    path: PathBuf,
    header: CacheHeader,
    /// In-memory copy of the full index table (grid_w × grid_h entries).
    index: Vec<IndexEntry>,
}

impl SdfCacheFile {
    /// Create a new, empty cache file. Overwrites any existing file at `path`.
    pub fn create(
        path: &Path,
        world_origin: (f32, f32),
        world_size: (f32, f32),
        tile_size: f32,
        sdf_resolution: u32,
        road_id_resolution: u32,
    ) -> io::Result<Self> {
        let grid_w = (world_size.0 / tile_size).ceil() as u32;
        let grid_h = (world_size.1 / tile_size).ceil() as u32;

        let header = CacheHeader {
            magic: MAGIC,
            version: VERSION,
            world_origin_x: world_origin.0,
            world_origin_y: world_origin.1,
            world_size_x: world_size.0,
            world_size_y: world_size.1,
            tile_size,
            sdf_resolution,
            road_id_resolution,
            grid_w,
            grid_h,
            _pad: 0,
        };

        let index_len = (grid_w * grid_h) as usize;
        let index = vec![IndexEntry { offset: 0, compressed_size: 0, _pad: 0 }; index_len];

        let mut file = File::create(path)?;
        file.write_all(bytemuck::bytes_of(&header))?;
        file.write_all(bytemuck::cast_slice(&index))?;
        file.flush()?;

        Ok(Self {
            path: path.to_owned(),
            header,
            index,
        })
    }

    /// Open an existing cache file and load its index into memory.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;

        let mut header = CacheHeader::zeroed();
        file.read_exact(bytemuck::bytes_of_mut(&mut header))?;

        if header.magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad cache magic"));
        }
        if header.version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported cache version {}", header.version),
            ));
        }

        let index_len = (header.grid_w * header.grid_h) as usize;
        let mut index = vec![IndexEntry { offset: 0, compressed_size: 0, _pad: 0 }; index_len];
        file.read_exact(bytemuck::cast_slice_mut(&mut index))?;

        Ok(Self {
            path: path.to_owned(),
            header,
            index,
        })
    }

    /// Convert a TileKey to a grid index, if it falls within this cache's grid.
    fn tile_to_grid_index(&self, key: TileKey) -> Option<usize> {
        let origin_ix = (self.header.world_origin_x / self.header.tile_size).floor() as i32;
        let origin_iy = (self.header.world_origin_y / self.header.tile_size).floor() as i32;
        let gx = key.ix - origin_ix;
        let gy = key.iy - origin_iy;
        if gx < 0 || gy < 0 || gx >= self.header.grid_w as i32 || gy >= self.header.grid_h as i32 {
            return None;
        }
        Some((gy as u32 * self.header.grid_w + gx as u32) as usize)
    }

    /// Check if a tile is present in the cache.
    pub fn has_tile(&self, key: TileKey) -> bool {
        self.tile_to_grid_index(key)
            .map(|i| self.index[i].offset != 0)
            .unwrap_or(false)
    }

    /// Read and decompress a tile from disk. Returns None if the tile is not cached.
    pub fn read_tile(&self, key: TileKey) -> io::Result<Option<TileData>> {
        let idx = match self.tile_to_grid_index(key) {
            Some(i) => i,
            None => return Ok(None),
        };
        let entry = self.index[idx];
        if entry.offset == 0 {
            return Ok(None);
        }

        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(entry.offset))?;
        let mut compressed = vec![0u8; entry.compressed_size as usize];
        file.read_exact(&mut compressed)?;

        let decompressed = zstd::decode_all(&compressed[..])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let sdf_size = self.sdf_tile_bytes();
        let rid_size = self.road_id_tile_bytes();
        let expected = sdf_size + rid_size;
        if decompressed.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tile data size mismatch: got {} expected {}",
                    decompressed.len(),
                    expected
                ),
            ));
        }

        Ok(Some(TileData {
            sdf_pixels: decompressed[..sdf_size].to_vec(),
            road_id_pixels: decompressed[sdf_size..].to_vec(),
        }))
    }

    /// Compress and write a tile to the cache. Appends data to end of file
    /// and updates the in-memory index + on-disk index entry.
    pub fn write_tile(
        &mut self,
        key: TileKey,
        sdf_data: &[u8],
        road_id_data: &[u8],
    ) -> io::Result<()> {
        let idx = match self.tile_to_grid_index(key) {
            Some(i) => i,
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "tile outside cache grid",
                ))
            }
        };

        // Concatenate then compress
        let mut raw = Vec::with_capacity(sdf_data.len() + road_id_data.len());
        raw.extend_from_slice(sdf_data);
        raw.extend_from_slice(road_id_data);
        let compressed = zstd::encode_all(&raw[..], 3)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let mut file = OpenOptions::new().read(true).write(true).open(&self.path)?;

        // Append compressed data at end of file
        let offset = file.seek(SeekFrom::End(0))?;
        file.write_all(&compressed)?;

        // Update in-memory index
        self.index[idx] = IndexEntry {
            offset,
            compressed_size: compressed.len() as u32,
            _pad: 0,
        };

        // Write updated index entry on disk
        let header_size = std::mem::size_of::<CacheHeader>() as u64;
        let entry_size = std::mem::size_of::<IndexEntry>() as u64;
        let entry_offset = header_size + idx as u64 * entry_size;
        file.seek(SeekFrom::Start(entry_offset))?;
        file.write_all(bytemuck::bytes_of(&self.index[idx]))?;
        file.flush()?;

        Ok(())
    }

    /// Uncompressed SDF pixel bytes per tile.
    fn sdf_tile_bytes(&self) -> usize {
        let res = self.header.sdf_resolution as usize;
        res * res * 4 // RG16F = 2 channels × 2 bytes
    }

    /// Uncompressed road-ID pixel bytes per tile.
    fn road_id_tile_bytes(&self) -> usize {
        let res = self.header.road_id_resolution as usize;
        res * res * 2 // R16_SFLOAT = 1 channel × 2 bytes
    }

    /// Invalidate a tile in the cache (mark as not present).
    /// Zeros the index entry in memory and on disk. The compressed data
    /// remains on disk as dead space until a future compaction.
    pub fn invalidate_tile(&mut self, key: TileKey) -> io::Result<()> {
        let idx = match self.tile_to_grid_index(key) {
            Some(i) => i,
            None => return Ok(()), // outside grid — nothing to do
        };
        if self.index[idx].offset == 0 {
            return Ok(()); // already absent
        }

        self.index[idx] = IndexEntry { offset: 0, compressed_size: 0, _pad: 0 };

        let mut file = OpenOptions::new().read(true).write(true).open(&self.path)?;
        let header_size = std::mem::size_of::<CacheHeader>() as u64;
        let entry_size = std::mem::size_of::<IndexEntry>() as u64;
        let entry_offset = header_size + idx as u64 * entry_size;
        file.seek(SeekFrom::Start(entry_offset))?;
        file.write_all(bytemuck::bytes_of(&self.index[idx]))?;
        file.flush()?;
        Ok(())
    }

    pub fn grid_w(&self) -> u32 {
        self.header.grid_w
    }

    pub fn grid_h(&self) -> u32 {
        self.header.grid_h
    }
}

// ---------------------------------------------------------------------------
// TileLoadQueue — background I/O thread for async tile loading
// ---------------------------------------------------------------------------

/// A completed tile load ready for GPU upload.
pub struct TileLoadResult {
    pub key: TileKey,
    pub slot: u32,
    pub data: TileData,
}

/// Request sent to the I/O thread.
struct LoadRequest {
    key: TileKey,
    slot: u32,
}

/// Sentinel: the I/O thread exits when it receives `None`.
enum IoMessage {
    Load(LoadRequest),
    Shutdown,
}

pub struct TileLoadQueue {
    tx: mpsc::SyncSender<IoMessage>,
    rx: mpsc::Receiver<TileLoadResult>,
    handle: Option<thread::JoinHandle<()>>,
    inflight: HashSet<TileKey>,
}

impl TileLoadQueue {
    /// Spawn a background I/O thread that reads tiles from `cache_path`.
    pub fn new(cache_path: &Path) -> io::Result<Self> {
        let cache = SdfCacheFile::open(cache_path)?;
        let (req_tx, req_rx) = mpsc::sync_channel::<IoMessage>(MAX_INFLIGHT);
        let (res_tx, res_rx) = mpsc::channel::<TileLoadResult>();

        let handle = thread::Builder::new()
            .name("tile-io".into())
            .spawn(move || {
                Self::io_thread(cache, req_rx, res_tx);
            })
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(Self {
            tx: req_tx,
            rx: res_rx,
            handle: Some(handle),
            inflight: HashSet::new(),
        })
    }

    fn io_thread(
        cache: SdfCacheFile,
        rx: mpsc::Receiver<IoMessage>,
        tx: mpsc::Sender<TileLoadResult>,
    ) {
        while let Ok(msg) = rx.recv() {
            match msg {
                IoMessage::Shutdown => break,
                IoMessage::Load(req) => {
                    match cache.read_tile(req.key) {
                        Ok(Some(data)) => {
                            let _ = tx.send(TileLoadResult {
                                key: req.key,
                                slot: req.slot,
                                data,
                            });
                        }
                        Ok(None) => {
                            // Tile not in cache — silently skip (caller will GPU-generate)
                        }
                        Err(e) => {
                            log::warn!("tile cache read error for {:?}: {}", req.key, e);
                        }
                    }
                }
            }
        }
    }

    /// Submit a tile load request. Returns false if the queue is full or
    /// this tile is already in-flight.
    pub fn request(&mut self, key: TileKey, slot: u32) -> bool {
        if self.inflight.len() >= MAX_INFLIGHT || self.inflight.contains(&key) {
            return false;
        }
        match self.tx.try_send(IoMessage::Load(LoadRequest { key, slot })) {
            Ok(()) => {
                self.inflight.insert(key);
                true
            }
            Err(_) => false,
        }
    }

    /// Drain completed loads (up to `max` per call). Returns them for GPU upload.
    pub fn drain_completed(&mut self, max: usize) -> Vec<TileLoadResult> {
        let mut results = Vec::new();
        for _ in 0..max {
            match self.rx.try_recv() {
                Ok(result) => {
                    self.inflight.remove(&result.key);
                    results.push(result);
                }
                Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => break,
            }
        }
        results
    }

    /// Check if a tile is currently being loaded.
    pub fn is_loading(&self, key: &TileKey) -> bool {
        self.inflight.contains(key)
    }

    /// Shut down the background I/O thread.
    pub fn shutdown(&mut self) {
        let _ = self.tx.send(IoMessage::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for TileLoadQueue {
    fn drop(&mut self) {
        self.shutdown();
    }
}
