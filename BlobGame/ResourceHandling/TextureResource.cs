using Raylib_CsLo;

namespace BlobGame.ResourceHandling;

//public interface ITextureResource {
//    internal string Key { get; }
//}

/// <summary>
/// Game resource for textures.
/// </summary>
public class TextureResource {
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
    private Texture? _Reource { get; set; }
    /// <summary>
    /// The raylib resource. Returns the fallback if the resource has not been loaded (yet).
    /// </summary>
    public Texture Resource {
        get {
            if (_Reource == null)
                _Reource = ResourceRetriever(Key);

            return _Reource ?? Fallback;
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
        _Reource = resourceRetriever(key);
    }
}
