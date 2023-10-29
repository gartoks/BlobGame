using Raylib_CsLo;

namespace BlobGame.ResourceHandling;

/// <summary>
/// Game resource for sounds.
/// </summary>
internal sealed class SoundResource {
    /// <summary>
    /// The key of the resource.
    /// </summary>
    public string Key { get; }

    /// <summary>
    /// A fallback resource to use if the resource could not be loaded or is still loading.
    /// </summary>
    private Sound Fallback { get; }
    /// <summary>
    /// Function to retrieve the resource from the resource manager. Used to check if the resource has been loaded.
    /// </summary>
    private Func<string, Sound?> ResourceRetriever { get; }

    /// <summary>
    /// The raylib resource. Is null if the resource has not been loaded yet.
    /// </summary>
    private Sound? _Resource { get; set; }
    /// <summary>
    /// The raylib resource. Returns the fallback if the resource has not been loaded (yet).
    /// </summary>
    public Sound Resource {
        get {
            if (!IsLoaded())
                _Resource = ResourceRetriever(Key);

            return _Resource ?? Fallback;
        }
    }

    /// <summary>
    /// Constructor for a new sound resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="fallback"></param>
    /// <param name="resourceRetriever"></param>
    internal SoundResource(string key, Sound fallback, Func<string, Sound?> resourceRetriever) {
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
    internal bool IsLoaded(){
        return _Resource != null;
    }
}
