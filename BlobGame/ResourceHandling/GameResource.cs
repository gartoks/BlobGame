namespace BlobGame.ResourceHandling;

internal delegate object? ResourceRetrieverDelegate(string key);    // sadly this has to be object? and cannot be T? because delegates dont work properly when T is a struct

/// <summary>
/// Base class for Game resources.
/// </summary>
internal abstract class GameResource<T> {
    /// <summary>
    /// The key of the resource.
    /// </summary>
    public string Key { get; }

    /// <summary>
    /// A fallback resource to use if the resource could not be loaded or is still loading.
    /// </summary>
    private T Fallback { get; }
    /// <summary>
    /// Function to retrieve the resource from the resource manager. Used to check if the resource has been loaded.
    /// </summary>
    private ResourceRetrieverDelegate ResourceRetriever { get; }

    /// <summary>
    /// The raylib resource. Is null if the resource has not been loaded yet.
    /// </summary>
    private T? _Resource { get; set; }
    /// <summary>
    /// The raylib resource. Returns the fallback if the resource has not been loaded (yet).
    /// </summary>
    public T Resource {
        get {
            if (!IsLoaded) {
                _Resource = (T?)ResourceRetriever(Key);
                //Log.WriteLineIf(_Resource != null, $"Resource {Key} loaded.");
            }

            if (IsLoaded)
                return _Resource!;
            else
                return Fallback;
        }
    }

    /// <summary>
    /// Returns whether the resource is loaded yet.
    /// </summary>
    public bool IsLoaded => _Resource != null && !_Resource.Equals(default(T));

    /// <summary>
    /// Constructor for a new resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="fallback"></param>
    /// <param name="resourceRetriever"></param>
    protected GameResource(string key, T fallback, ResourceRetrieverDelegate resourceRetriever) {
        Key = key;

        Fallback = fallback;
        ResourceRetriever = resourceRetriever;
        _Resource = (T?)resourceRetriever(key);
    }

    /// <summary>
    /// Unloads the resource.
    /// </summary>
    internal void Unload() {
        _Resource = default;
    }

    public void WaitForLoad() {
        while (!IsLoaded) {
            T? r = Resource;
            Thread.Sleep(1);
        }
    }
}
