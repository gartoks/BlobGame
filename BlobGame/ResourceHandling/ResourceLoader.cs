using SimpleGL.Util;
using System.Collections.Concurrent;

namespace BlobGame.ResourceHandling;

public enum eResourceLoadStatus { NotLoaded, Loading, Loaded, Unloaded }

internal abstract class ResourceLoader {
    internal abstract void Load();
    internal abstract void Unload();
    public abstract eResourceLoadStatus GetResourceState(string key);
    public abstract void Load(params string[] keys);
    public abstract void Unload(params string[] keys);
    public abstract void ReloadAll();
    internal abstract void LoadResource(string key);
}

/// <summary>
/// Class for loading resources.
/// </summary>
/// <typeparam name="TResource"></typeparam>
/// <typeparam name="TGameResource"></typeparam>
internal abstract class ResourceLoader<TResource> : ResourceLoader {
    /// <summary>
    /// Mirror of the resource loading queue in <see cref="ResourceManager"/>.
    /// </summary>
    private BlockingCollection<(string key, Type type)> ResourceLoadingQueue { get; }

    /// <summary>
    /// Loaded resources.
    /// </summary>
    private ConcurrentDictionary<string, TResource> LoadedResources { get; }
    private ConcurrentDictionary<string, ManualResetEvent> LoadingResources { get; }

    internal event Action<string, TResource> ResourceLoaded;

    internal ResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue) {
        ResourceLoadingQueue = resourceLoadingQueue;

        LoadedResources = new();
        LoadingResources = new();
    }

    /// <summary>
    /// Loads the resource loader.
    /// </summary>
    internal override void Load() {
    }

    /// <summary>
    /// Unloads the resource loader.
    /// </summary>
    internal override void Unload() {
        foreach (string key in LoadingResources.Keys.ToList())
            Unload(key);

        foreach (string key in LoadedResources.Keys.ToList())
            Unload(key);

        LoadedResources.Clear();
        LoadingResources.Clear();
    }

    /// <summary>
    /// Returns whether the resource is loaded.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public override eResourceLoadStatus GetResourceState(string key) {
        if (LoadedResources.ContainsKey(key))
            return eResourceLoadStatus.Loaded;
        else if (LoadingResources.ContainsKey(key))
            return eResourceLoadStatus.Loading;
        else
            return eResourceLoadStatus.NotLoaded;
    }

    /// <summary>
    /// Gets the resource.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public TResource GetResource(string key) {
        if (LoadingResources.TryGetValue(key, out ManualResetEvent? loadWaiter))
            loadWaiter.WaitOne();

        if (LoadedResources.TryGetValue(key, out TResource? resouce))
            return resouce;

        throw new InvalidOperationException($"Resource {key} is not loaded or loading");
    }

    /// <summary>
    /// Attempts to get the resource.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public bool TryGetResource(string key, out TResource? resource) {
        return LoadedResources.TryGetValue(key, out resource);
    }

    /// <summary>
    /// Marks a resource for loading.
    /// </summary>
    /// <param name="key"></param>
    public override void Load(params string[] keys) {
        foreach (string key in keys) {
            if (GetResourceState(key) != eResourceLoadStatus.NotLoaded)
                return;

            LoadingResources[key] = new ManualResetEvent(false);
            ResourceLoadingQueue.Add((key, typeof(TResource)));
        }
    }

    /// <summary>
    /// Unloads the resource.
    /// </summary>
    /// <param name="key"></param>
    public override void Unload(params string[] keys) {    // TODO maybe internal
        foreach (string key in keys) {


            if (LoadingResources.TryRemove(key, out ManualResetEvent? loadWaiter)) {
                loadWaiter?.Set();
                return;
            }

            if (!LoadedResources.TryRemove(key, out TResource? res))
                return;

            UnloadResourceInternal(key, res);
        }
    }

    /// <summary>
    /// Reloads all resources
    /// </summary>
    public override void ReloadAll() {
        foreach (string key in LoadedResources.Keys.ToList()) {
            Unload(key);
            Load(key);
        }
    }

    /// <summary>
    /// Actually performs the loading of the resource.
    /// </summary>
    /// <param name="key"></param>
    internal override void LoadResource(string key) {
        // Resource is not added to "loaded" of it does not exist as a waiter.
        // This can happen if the resource is unloaded before it is finished loading.
        if (!LoadingResources.Remove(key, out ManualResetEvent? loadWaiter))
            return;

        TResource? resource = LoadResourceInternal(key);
        if (resource == null) {
            Log.WriteLine($"The {typeof(TResource).Name} resource file for {key} does not exist.", eLogType.Error);
            return;
        }

        LoadedResources[key] = resource;
        loadWaiter.Set();
        ResourceLoaded?.Invoke(key, resource);
    }

    /*
        /// <summary>
        /// Creates a new game resource object.
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        protected TGameResource Create(string key) {
            ResourceRetrieverDelegate del = key => TryGetResource(key);
            return (TGameResource?)Activator.CreateInstance(typeof(TGameResource), BindingFlags.NonPublic | BindingFlags.Instance, null, new object[] { key, _Fallback, del }, CultureInfo.InvariantCulture)!;
        }
    */
    /// <summary>
    /// Performs the actual loading of the resource.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    protected abstract TResource? LoadResourceInternal(string key);

    /// <summary>
    /// Unloads the actual resource.
    /// </summary>
    /// <param name="key">The key of the resource</param>
    protected abstract void UnloadResourceInternal(string key, TResource resource);
}