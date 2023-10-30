using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.ResourceHandling;
/// <summary>
/// Game resource for fonts.
/// </summary>
internal sealed class FontResource {
    /// <summary>
    /// The key of the resource.
    /// </summary>
    public string Key { get; }

    /// <summary>
    /// A fallback resource to use if the resource could not be loaded or is still loading.
    /// </summary>
    private Font Fallback { get; }
    /// <summary>
    /// Function to retrieve the resource from the resource manager. Used to check if the resource has been loaded.
    /// </summary>
    private Func<string, Font?> ResourceRetriever { get; }

    /// <summary>
    /// The raylib resource. Is null if the resource has not been loaded yet.
    /// </summary>
    private Font? _Resource { get; set; }
    /// <summary>
    /// The raylib resource. Returns the fallback if the resource has not been loaded (yet).
    /// </summary>
    public Font Resource {
        get {
            if (!IsLoaded())
                _Resource = ResourceRetriever(Key);

            return _Resource ?? Fallback;
        }
    }

    /// <summary>
    /// Constructor for a new font resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="fallback"></param>
    /// <param name="resourceRetriever"></param>
    internal FontResource(string key, Font fallback, Func<string, Font?> resourceRetriever) {
        Key = key;

        Fallback = fallback;
        ResourceRetriever = resourceRetriever;
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

    internal void Draw(string text, float fontSize, ColorResource tint, Vector2 position, /*Vector2? pivot = null, */float rotation = 0) {
        Draw(text, fontSize, tint.Resource, position, /*pivot, */rotation);
    }
    internal void Draw(string text, float fontSize, Color tint, Vector2 position, /*Vector2? pivot = null, */float rotation = 0) {
        /*if (pivot == null)
            pivot = Vector2.Zero;
*/
        Raylib.DrawTextPro(
            Resource,
            text,
            position,
            new Vector2(),
            rotation,
            fontSize,
            fontSize / 16f,
            tint);
    }
}
