using BlobGame.ResourceHandling;
using SimpleGL.Util;

namespace BlobGame.Util;
internal class ResourceThreadBase : ThreadBase {
    public ResourceThreadBase(int ticksPerSecond)
        : base("ResourceThread", ticksPerSecond) {
    }

    protected override void OnStart() {
        ResourceManager.Initialize();
        ResourceManager.Load();
    }

    protected override void OnStop() {
        ResourceManager.Unload();
    }

    protected override void Run(float deltaTime) {
        ResourceManager.Update();
    }
}
