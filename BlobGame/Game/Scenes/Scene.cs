namespace BlobGame.Game.Scenes;
internal abstract class Scene {
    internal virtual void Load() { }

    internal virtual void Update(float dT) { }

    internal virtual void Draw() { }

    internal virtual void Unload() { }

}
