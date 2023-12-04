using SimpleGL.Graphics.Textures;
using System.Collections.Concurrent;

namespace BlobGame.ResourceHandling.Resources;
/*
/// <summary>
/// Game resource for textures.
/// </summary>
internal sealed class TextureResource : GameResource<Texture> {


    /// <summary>
    /// Constructor for a new texture resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="fallback"></param>
    /// <param name="resourceRetriever"></param>
    internal TextureResource(string key, Texture fallback, ResourceRetrieverDelegate resourceRetriever)
        : base(key, fallback, resourceRetriever) {
    }

    public TextureResource(string key, TextureResourceLoader loader, Texture? resource)
        : base(key, loader, resource) {
    }

    internal void Draw(Vector2 position, Vector2? pivot = null, Vector2? scale = null, float rotation = 0, Color4? tint = null) {
        if (scale == null)
            scale = Vector2.One;

        if (pivot == null)
            pivot = Vector2.Zero;

        float w = Resource.Width * scale.Value.X;
        float h = Resource.Height * scale.Value.Y;

        Raylib.DrawTexturePro(
                    Resource,
                    new Rectangle(0, 0, Resource.Width, Resource.Height),
                    new Rectangle(position.X, position.Y, w, h),
                    new Vector2(w * pivot.Value.X, h * pivot.Value.Y),
                    rotation,
                    tint != null ? tint.Value : Color4.wh);
    }

    internal void Draw(Box2 bounds, Vector2? pivot = null, float rotation = 0, Color4? tint = null) {
        if (pivot == null)
            pivot = Vector2.Zero;

        Raylib.DrawTexturePro(
                    Resource,
                    new Rectangle(0, 0, Resource.width, Resource.height),
                    bounds,
                    new Vector2(bounds.width * pivot.Value.X, bounds.height * pivot.Value.Y),
                    rotation,
                    tint != null ? tint.Value : Color4.White);
    }
}*/

internal sealed class TextureResourceLoader : ResourceLoader<Texture2D> {
    public TextureResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue)
        : base(resourceLoadingQueue) {
    }

    protected override Texture2D LoadResourceInternal(string key) {
        Texture2D? res = ResourceManager.MainTheme.LoadTexture(key) ?? ResourceManager.DefaultTheme.LoadTexture(key);
        return res ?? throw new ArgumentException($"Texture resource '{key}' does not exist in theme.");
    }

    protected override void UnloadResourceInternal(string key, Texture2D texture) {
        texture.Dispose();
    }
}
