using BlobGame.App;
using BlobGame.Audio;
using BlobGame.ResourceHandling.Resources;
using OpenTK.Mathematics;
using SimpleGL.Graphics.Textures;
using SimpleGL.Util;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace BlobGame.ResourceHandling;
/// <summary>
/// Class for managing the game's resources. Handles loading and caching of resources.
/// </summary>
internal static class ResourceManager {
    /// <summary>
    /// Time in milliseconds to wait for a resource to be needed to be loaded before continuing the frame.
    /// </summary>
    private const int RESOURCE_LOADING_TIMEOUT = 0;
    /// <summary>
    /// Thread-safe queue of resources to be loaded.
    /// </summary>
    private static BlockingCollection<(string key, Type type)> ResourceLoadingQueue { get; }

    public static ColorResourceLoader ColorLoader { get; }
    public static FontResourceLoader FontLoader { get; }
    public static TextureResourceLoader TextureLoader { get; }
    public static SoundResourceLoader SoundLoader { get; }
    public static MusicResourceLoader MusicLoader { get; }
    public static TextResourceLoader TextLoader { get; }
    public static NPatchTextureResourceLoader NPatchLoader { get; }
    private static ResourceLoader[] ResourceLoaders { get; }

    /// <summary>
    /// Base theme.
    /// </summary>
    internal static Theme DefaultTheme { get; }
    /// <summary>
    /// Additional theme.
    /// </summary>
    internal static Theme MainTheme { get; private set; }

    public static event Action<string, Type> ResourceLoaded;

    /// <summary>
    /// Static constructor to initialize the resource loading queue and other required properties.
    /// </summary>
    static ResourceManager() {
        ResourceLoadingQueue = new BlockingCollection<(string key, Type type)>();

        ColorLoader = new(ResourceLoadingQueue);
        FontLoader = new(ResourceLoadingQueue);
        TextureLoader = new(ResourceLoadingQueue);
        SoundLoader = new(ResourceLoadingQueue);
        MusicLoader = new(ResourceLoadingQueue);
        TextLoader = new(ResourceLoadingQueue);
        NPatchLoader = new(ResourceLoadingQueue);

        ResourceLoaders = new ResourceLoader[] {
            ColorLoader,
            FontLoader,
            TextureLoader,
            SoundLoader,
            MusicLoader,
            TextLoader,
            NPatchLoader
        };

        ColorLoader.ResourceLoaded += (key, resource) => ResourceLoaded?.Invoke(key, typeof(Color4));
        FontLoader.ResourceLoaded += (key, resource) => ResourceLoaded?.Invoke(key, typeof(FontFamilyData));
        TextureLoader.ResourceLoaded += (key, resource) => ResourceLoaded?.Invoke(key, typeof(Texture2D));
        SoundLoader.ResourceLoaded += (key, resource) => ResourceLoaded?.Invoke(key, typeof(Sound));
        MusicLoader.ResourceLoaded += (key, resource) => ResourceLoaded?.Invoke(key, typeof(Music));
        TextLoader.ResourceLoaded += (key, resource) => ResourceLoaded?.Invoke(key, typeof(IReadOnlyDictionary<string, string>));
        NPatchLoader.ResourceLoaded += (key, resource) => ResourceLoaded?.Invoke(key, typeof(NPatchTexture));

        DefaultTheme = new Theme(Files.GetResourceFilePath("MelbaToast.theme"));
        MainTheme = DefaultTheme;
    }

    /// <summary>
    /// Used to initialize the resource manager. Currently does nothing.
    /// </summary>
    internal static void Initialize() {
    }

    /// <summary>
    /// Loads default resources.
    /// </summary>
    internal static void Load() {
        DefaultTheme.Load();

        foreach (ResourceLoader loader in ResourceLoaders)
            loader.Load();
    }

    /// <summary>
    /// Unloads all resources.
    /// </summary>
    internal static void Unload() {
        foreach (ResourceLoader loader in ResourceLoaders)
            loader.Unload();

        DefaultTheme?.Unload();
        DefaultTheme?.Dispose();

        MainTheme?.Unload();
        MainTheme?.Dispose();
    }

    /// <summary>
    /// Returns whether the resource is loaded.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public static eResourceLoadStatus GetResourceState(string key) {
        foreach (ResourceLoader loader in ResourceLoaders) {
            if (loader.GetResourceState(key) is eResourceLoadStatus.Loading or eResourceLoadStatus.Loaded)
                return loader.GetResourceState(key);
        }

        return eResourceLoadStatus.NotLoaded;
    }

    /// <summary>
    /// Clears the resource cache and reloads everything.
    /// </summary>
    private static void ReloadResources() {
        while (ResourceLoadingQueue.TryTake(out _)) ;

        Log.WriteLine("Reloading resources.");
        foreach (ResourceLoader loader in ResourceLoaders)
            loader.ReloadAll();
    }

    /// <summary>
    /// Called every frame. Checks the resource loading queue and loads resources if needed.
    /// </summary>
    internal static void Update() {
        while (ResourceLoadingQueue.TryTake(out (string key, Type type) resource, RESOURCE_LOADING_TIMEOUT)) {
            //Log.WriteLine($"Loading resource {resource.key} of type {resource.type}");
            LoadResource(resource.key, resource.type);
        }
    }

    /// <summary>
    /// Loads a resource from the given key and type.
    /// </summary>
    /// <param name="key">The key of the resource.</param>
    /// <param name="type">The type of the raylib resource type</param>
    private static void LoadResource(string key, Type type) {
        if (type == typeof(Color4)) {
            ColorLoader.LoadResource(key);
        } else if (type == typeof(FontFamilyData)) {
            FontLoader.LoadResource(key);
        } else if (type == typeof(Texture2D)) {
            TextureLoader.LoadResource(key);
        } else if (type == typeof(Sound)) {
            SoundLoader.LoadResource(key);
        } else if (type == typeof(Music)) {
            MusicLoader.LoadResource(key);
        } else if (type == typeof(IReadOnlyDictionary<string, string>)) {
            TextLoader.LoadResource(key);
        } else if (type == typeof(NPatchTexture)) {
            NPatchLoader.LoadResource(key);
        } else {
            Debug.WriteLine($"Resource type {type} is not supported");
        }
    }

    /// <summary>
    /// Sets the main theme to the one named
    /// </summary>
    /// <param name="name"></param>
    public static void SetTheme(string name) {
        if (MainTheme?.Name == name)
            return;

        string filename = Files.GetResourceFilePath(name + ".theme");

        if (!File.Exists(filename)) {
            Debug.WriteLine($"ERROR: Theme named '{name}' doesn't exist. Using Fallback.");
            return;
        }

        if (MainTheme != DefaultTheme) {
            MainTheme.Unload();
            MainTheme.Dispose();
        }

        MainTheme = new Theme(filename);
        MainTheme.Load();

        // force everything to reload
        ReloadResources();
    }
}
