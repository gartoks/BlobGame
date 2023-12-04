/*namespace BlobGame.ResourceHandling;

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
    /// The raylib resource. Is null if the resource has not been loaded yet.
    /// </summary>
    private T? _Resource { get; set; }
    /// <summary>
    /// The raylib resource. Returns the fallback if the resource has not been loaded (yet).
    /// </summary>
    public T Resource {
        get {
            eResourceLoadStatus status = Status;
            if (status is eResourceLoadStatus.NotLoaded or eResourceLoadStatus.Unloaded)
                throw new InvalidOperationException($"Resource {Key} is not loaded");

            if (status == eResourceLoadStatus.Loading)
                ResourceLoadWaiter?.WaitOne();

            return _Resource!;
        }
    }

    /// <summary>
    /// Returns the load status of the resource.
    /// </summary>
    public eResourceLoadStatus Status => ResourceManager.GetResourceState(Key);

    private ManualResetEvent? ResourceLoadWaiter { get; }

    /// <summary>
    /// Constructor for a new resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="fallback"></param>
    /// <param name="resourceRetriever"></param>
    protected GameResource(string key, T? resource) {
        Key = key;

        if (resource != null && !resource.Equals(default(T))) {
            _Resource = resource;
        } else {
            ResourceLoadWaiter = new ManualResetEvent(false);
        }
    }

    internal void SetResource(T resource) {
        _Resource = resource;
        ResourceLoadWaiter?.Set();
    }

    /// <summary>
    /// Unloads the resource.
    /// </summary>
    internal void Unload() {
        _Resource = default;
    }
}
*/