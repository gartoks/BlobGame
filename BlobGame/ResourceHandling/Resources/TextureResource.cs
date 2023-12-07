using Raylib_CsLo;
using System.Collections.Concurrent;
using System.Numerics;

namespace BlobGame.ResourceHandling.Resources;
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

    internal void Draw(Vector2 position, Vector2? pivot = null, Vector2? scale = null, float rotation = 0, Color? tint = null) {
        if (scale == null)
            scale = Vector2.One;

        if (pivot == null)
            pivot = Vector2.Zero;

        float w = Resource.width * scale.Value.X;
        float h = Resource.height * scale.Value.Y;

        Raylib.DrawTexturePro(
                    Resource,
                    new Rectangle(0, 0, Resource.width, Resource.height),
                    new Rectangle(position.X, position.Y, w, h),
                    new Vector2(w * pivot.Value.X, h * pivot.Value.Y),
                    rotation,
                    tint != null ? tint.Value : Raylib.WHITE);
    }

    internal void Draw(Rectangle bounds, Vector2? pivot = null, float rotation = 0, Color? tint = null) {
        if (pivot == null)
            pivot = Vector2.Zero;

        Raylib.DrawTexturePro(
                    Resource,
                    new Rectangle(0, 0, Resource.width, Resource.height),
                    bounds,
                    new Vector2(bounds.width * pivot.Value.X, bounds.height * pivot.Value.Y),
                    rotation,
                    tint != null ? tint.Value : Raylib.WHITE);
    }
}

internal sealed class TextureResourceLoader : ResourceLoader<Texture, TextureResource> {
    public TextureResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue)
        : base(resourceLoadingQueue) {
    }

    protected override bool ResourceExistsInternal(string key) {
        return ResourceManager.MainTheme.DoesTextureExist(key);
    }

    protected override Texture LoadResourceInternal(string key) {
        Texture? res = ResourceManager.MainTheme.LoadTexture(key) ?? ResourceManager.DefaultTheme.LoadTexture(key);
        return res ?? Fallback.Resource;
    }

    protected override void UnloadResourceInternal(TextureResource resource) {
        if (resource.Resource.id != 0 && resource.Resource.id != Fallback.Resource.id)
            Raylib.UnloadTexture(resource.Resource);
    }
}
