use wre::ApplicationBuilder;

struct RoadApp;

fn main() {
    ApplicationBuilder::new(RoadApp).run().unwrap();
}
