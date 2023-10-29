using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.ResourceHandling;

//public interface ITextureResource {
//    internal string Key { get; }
//}

/// <summary>
/// Game resource for textures.
/// </summary>
internal sealed class TextureResource {
    /// <summary>
    /// The key of the resource.
    /// </summary>
    public string Key { get; }

    /// <summary>
    /// A fallback resource to use if the resource could not be loaded or is still loading.
    /// </summary>
    private Texture Fallback { get; }
    /// <summary>
    /// Function to retrieve the resource from the resource manager. Used to check if the resource has been loaded.
    /// </summary>
    private Func<string, Texture?> ResourceRetriever { get; }

    /// <summary>
    /// The raylib resource. Is null if the resource has not been loaded yet.
    /// </summary>
    private Texture? _Resource { get; set; }
    /// <summary>
    /// The raylib resource. Returns the fallback if the resource has not been loaded (yet).
    /// </summary>
    public Texture Resource {
        get {
            if (!IsLoaded())
                _Resource = ResourceRetriever(Key);

            return _Resource ?? Fallback;
        }
    }

    /// <summary>
    /// Constructor for a new texture resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="fallback"></param>
    /// <param name="resourceRetriever"></param>
    internal TextureResource(string key, Texture fallback, Func<string, Texture?> resourceRetriever) {
        Key = key;

        ResourceRetriever = resourceRetriever;
        Fallback = fallback;
        _Resource = resourceRetriever(key);
    }

    /// <summary>
    /// Unloads the resource.
    /// </summary>
    internal void Unload() {
        _Resource = null;
    }

    /// <summary>
    /// Returns whether the resource is loaded yet.
    /// </summary>
    internal bool IsLoaded() {
        return _Resource != null;
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
}
