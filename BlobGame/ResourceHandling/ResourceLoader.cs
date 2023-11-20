using BlobGame.Util;
using System.Collections.Concurrent;
using System.Globalization;
using System.Reflection;

namespace BlobGame.ResourceHandling;
/// <summary>
/// Class for loading resources.
/// </summary>
/// <typeparam name="TResource"></typeparam>
/// <typeparam name="TGameResource"></typeparam>
internal abstract class ResourceLoader<TResource, TGameResource> where TGameResource : GameResource<TResource> {
    /// <summary>
    /// Mirror of the resource loading queue in <see cref="ResourceManager"/>.
    /// </summary>
    private BlockingCollection<(string key, Type type)> ResourceLoadingQueue { get; }

    /// <summary>
    /// Loaded resources.
    /// </summary>
    private ConcurrentDictionary<string, (TResource? baseResource, TGameResource resource)> Resources { get; }

    /// <summary>
    /// Fall back resource. Used in case a resource has not yet been loaded.
    /// </summary>
    private TResource _Fallback { get; set; }
    /// <summary>
    /// The fallback game resource.
    /// </summary>
    internal TGameResource Fallback { get; private set; }

    internal ResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue) {
        ResourceLoadingQueue = resourceLoadingQueue;

        Resources = new ConcurrentDictionary<string, (TResource? baseResource, TGameResource resource)>();
    }

    /// <summary>
    /// Loads the resource loader.
    /// </summary>
    /// <param name="fallbackResource"></param>
    internal void Load(TResource fallbackResource) {
        _Fallback = fallbackResource;
        Fallback = Create("fallback");
    }

    /// <summary>
    /// Unloads the resource loader.
    /// </summary>
    internal void Unload() {
        foreach (KeyValuePair<string, (TResource? baseResource, TGameResource resource)> item in Resources) {
            Unload(item.Key);
        }
        Resources.Clear();
    }

    /// <summary>
    /// Actually performs the loading of the resource.
    /// </summary>
    /// <param name="key"></param>
    internal void LoadResource(string key) {
        if (!Resources.TryGetValue(key, out (TResource? baseResource, TGameResource resource) val)) {
            Log.WriteLine($"Unable to load {typeof(TResource).Name} resource '{key}'.", eLogType.Error);
            return;
        }

        TResource? resource = LoadResourceInternal(key);
        if (resource == null) {
            Log.WriteLine($"The {typeof(TResource).Name} resource file for {key} does not exist.", eLogType.Error);
            return;
        }

        Resources[key] = (resource!, val.resource);
    }

    /// <summary>
    /// Returns whether the resource is loaded.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public bool IsLoaded(string key) {
        return Resources.Any(pair => pair.Key == key && pair.Value.resource.IsLoaded);
    }

    /// <summary>
    /// Returns whether the resource is loading.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public bool IsLoading(string key) {
        return ResourceLoadingQueue.Any(r => r.key == key);
    }

    /// <summary>
    /// Gets the resource.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public TGameResource Get(string key) {
        Load(key);
        return Resources[key].resource;
    }

    /// <summary>
    /// Marks a resource for loading.
    /// </summary>
    /// <param name="key"></param>
    public void Load(string key) {    // TODO maybe internal
        if (!IsLoaded(key) && !IsLoading(key)) {
            TGameResource gameResource = Create(key);
            Resources[key] = (default, gameResource);
            ResourceLoadingQueue.Add((key, typeof(TResource)));
        }
    }

    /// <summary>
    /// Attempts to get the resource.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    private TResource? TryGetResource(string key) {
        if (Resources.TryGetValue(key, out (TResource? baseResource, TGameResource resource) texture))
            return texture.baseResource;

        return default;
    }

    /// <summary>
    /// Unloads the resource.
    /// </summary>
    /// <param name="key"></param>
    public void Unload(string key) {    // TODO maybe internal
        if (!Resources.TryGetValue(key, out (TResource? bR, TGameResource r) res))
            return;

        TGameResource resource = res.r;
        UnloadResourceInternal(resource);
        resource.Unload();
        Resources[key] = (default, resource);
    }

    /// <summary>
    /// Reloads all resources
    /// </summary>
    public void ReloadAll() {
        foreach (string key in Resources.Keys.ToList()) {
            Unload(key);
            Load(key);
        }
    }

    /// <summary>
    /// Creates a new game resource object.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    protected TGameResource Create(string key) {
        ResourceRetrieverDelegate del = key => TryGetResource(key);
        return (TGameResource?)Activator.CreateInstance(typeof(TGameResource), BindingFlags.NonPublic | BindingFlags.Instance, null, new object[] { key, _Fallback, del }, CultureInfo.InvariantCulture)!;
    }

    /// <summary>
    /// Performs the actual loading of the resource.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    protected abstract TResource? LoadResourceInternal(string key);

    /// <summary>
    /// Unloads the actual resource.
    /// </summary>
    /// <param name="resource"></param>
    protected abstract void UnloadResourceInternal(TGameResource resource);
}