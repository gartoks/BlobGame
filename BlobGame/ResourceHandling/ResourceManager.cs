using BlobGame.App;
using Raylib_CsLo;
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

    /// <summary>
    /// Font resources.
    /// </summary>
    private static ConcurrentDictionary<string, Font> Fonts { get; }
    /// <summary>
    /// Texture resources.
    /// </summary>
    private static ConcurrentDictionary<string, Texture> Textures { get; }
    /// <summary>
    /// Sound resources.
    /// </summary>
    private static ConcurrentDictionary<string, Sound> Sounds { get; }
    /// <summary>
    /// Sound resources.
    /// </summary>
    private static ConcurrentDictionary<string, Music> Music { get; }

    /// <summary>
    /// Default raylib font.
    /// </summary>
    private static Font _DefaultFont { get; set; }
    /// <summary>
    /// Default fallback font resource.
    /// </summary>
    public static FontResource DefaultFont { get; private set; }

    /// <summary>
    /// Default raylib texture.
    /// </summary>
    private static Texture _DefaultTexture { get; set; }
    /// <summary>
    /// Default fallback texture resource.
    /// </summary>
    public static TextureResource DefaultTexture { get; private set; }

    /// <summary>
    /// Default raylib sound.
    /// </summary>
    private static Sound _DefaultSound { get; set; }
    /// <summary>
    /// Default fallback sound resource.
    /// </summary>
    public static SoundResource DefaultSound { get; private set; }

    /// <summary>
    /// Default raylib music.
    /// </summary>
    private static Music _DefaultMusic { get; set; }
    /// <summary>
    /// Default fallback music resource.
    /// </summary>
    public static MusicResource DefaultMusic { get; private set; }

    /// <summary>
    /// Static constructor to initialize the resource loading queue and other required properties.
    /// </summary>
    static ResourceManager() {
        ResourceLoadingQueue = new BlockingCollection<(string key, Type type)>();

        Fonts = new ConcurrentDictionary<string, Font>();
        Textures = new ConcurrentDictionary<string, Texture>();
        Sounds = new ConcurrentDictionary<string, Sound>();
        Music = new ConcurrentDictionary<string, Music>();
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
        _DefaultFont = Raylib.GetFontDefault();
        Image image = Raylib.GenImageColor(1, 1, Raylib.BLANK);
        _DefaultTexture = Raylib.LoadTextureFromImage(image);
        _DefaultSound = new Sound();
        _DefaultMusic = new Music();

        DefaultFont = new FontResource("default", _DefaultFont, TryGetFont);
        DefaultTexture = new TextureResource("default", _DefaultTexture, TryGetTexture);
        DefaultSound = new SoundResource("default", _DefaultSound, TryGetSound);
        DefaultMusic = new MusicResource("default", _DefaultMusic, TryGetMusic);
    }

    /// <summary>
    /// Called every frame. Checks the resource loading queue and loads resources if needed.
    /// </summary>
    internal static void Update() {
        while (ResourceLoadingQueue.TryTake(out (string key, Type type) resource, RESOURCE_LOADING_TIMEOUT)) {
            LoadResource(resource.key, resource.type);
        }
    }

    /// <summary>
    /// Loads a resource from the given key and type.
    /// </summary>
    /// <param name="key">The key of the resource.</param>
    /// <param name="type">The type of the raylib resource type</param>
    private static void LoadResource(string key, Type type) {
        if (type == typeof(Font)) {
            if (Fonts.ContainsKey(key))
                return;

            string path = Files.GetResourceFilePath("Fonts", $"{key}.ttf");

            Font tmpFont = Raylib.LoadFont(path);
            Font font = Raylib.LoadFontEx(path, 200, tmpFont.glyphCount);

            if (font.texture.id == 0) {
                Debug.WriteLine($"Failed to load font {key} from {path}");
                return;
            }

            if (!Fonts.TryAdd(key, font))
                Debug.WriteLine($"Failed to add font {key} to dictionary");
        } else if (type == typeof(Texture)) {
            if (Textures.ContainsKey(key))
                return;

            string path = Files.GetResourceFilePath("Textures", $"{key}.png");

            Texture texture = Raylib.LoadTexture(path);

            if (texture.id == 0) {
                Debug.WriteLine($"Failed to load texture {key} from {path}");
                return;
            }

            if (!Textures.TryAdd(key, texture))
                Debug.WriteLine($"Failed to add texture {key} to dictionary");
        } else if (type == typeof(Sound)) {
            if (Sounds.ContainsKey(key))
                return;

            string path = Files.GetResourceFilePath("Sounds", $"{key}.wav");

            Sound sound = Raylib.LoadSound(path);

            /*if (sound. == 0) {
                Debug.WriteLine($"Failed to load sound {key} from {path}");
                return;
            }*/

            if (!Sounds.TryAdd(key, sound))
                Debug.WriteLine($"Failed to add sound {key} to dictionary");
        } else if (type == typeof(Music)) {
            if (Music.ContainsKey(key))
                return;

            string path = Files.GetResourceFilePath("Music", $"{key}.wav");

            Music music = Raylib.LoadMusicStream(path);

            /*if (sound. == 0) {
                Debug.WriteLine($"Failed to load sound {key} from {path}");
                return;
            }*/

            if (!Music.TryAdd(key, music))
                Debug.WriteLine($"Failed to add music {key} to dictionary");
        } else {
            Debug.WriteLine($"Resource type {type} is not supported");
        }
    }

    /// <summary>
    /// Queues a font for loading from the given key.
    /// </summary>
    /// <param name="key"></param>
    public static void LoadFont(string key) {
        if (!Fonts.ContainsKey(key) && !ResourceLoadingQueue.Any(r => r.key == key))
            ResourceLoadingQueue.Add((key, typeof(Font)));
    }

    /// <summary>
    /// Gets a font resource from the given key. Queues it for loading if needed.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public static FontResource GetFont(string key) {
        LoadFont(key);
        return new FontResource(key, _DefaultFont, TryGetFont);
    }

    /// <summary>
    /// Tries to get a raylib font from the given key. Returns null if it doesn't exist.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    private static Font? TryGetFont(string key) {
        if (Fonts.TryGetValue(key, out Font font))
            return font;

        return null;
    }

    /// <summary>
    /// Queues a texture for loading from the given key.
    /// </summary>
    /// <param name="key"></param>
    public static void LoadTexture(string key) {
        if (!Textures.ContainsKey(key))
            ResourceLoadingQueue.Add((key, typeof(Texture)));
    }

    /// <summary>
    /// Gets a texture resource from the given key. Queues it for loading if needed.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public static TextureResource GetTexture(string key) {
        LoadTexture(key);
        return new TextureResource(key, _DefaultTexture, TryGetTexture);
    }

    /// <summary>
    /// Tries to get a raylib texture from the given key. Returns null if it doesn't exist.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    private static Texture? TryGetTexture(string key) {
        if (Textures.TryGetValue(key, out Texture texture))
            return texture;

        return null;
    }

    /// <summary>
    /// Queues a sound for loading from the given key.
    /// </summary>
    /// <param name="key"></param>
    public static void LoadSound(string key) {
        if (!Sounds.ContainsKey(key))
            ResourceLoadingQueue.Add((key, typeof(Sound)));
    }

    /// <summary>
    /// Gets a sound resource from the given key. Queues it for loading if needed.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public static SoundResource GetSound(string key) {
        LoadSound(key);
        return new SoundResource(key, _DefaultSound, TryGetSound);
    }

    /// <summary>
    /// Tries to get a raylib sound from the given key. Returns null if it doesn't exist.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    private static Sound? TryGetSound(string key) {
        if (Sounds.TryGetValue(key, out Sound sound))
            return sound;

        return null;
    }

    /// <summary>
    /// Queues a Music for loading from the given key.
    /// </summary>
    /// <param name="key"></param>
    public static void LoadMusic(string key) {
        if (!Music.ContainsKey(key))
            ResourceLoadingQueue.Add((key, typeof(Music)));
    }

    /// <summary>
    /// Gets a Music resource from the given key. Queues it for loading if needed.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public static MusicResource GetMusic(string key) {
        LoadMusic(key);
        return new MusicResource(key, _DefaultMusic, TryGetMusic);
    }

    /// <summary>
    /// Tries to get a raylib Music from the given key. Returns null if it doesn't exist.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    private static Music? TryGetMusic(string key) {
        if (Music.TryGetValue(key, out Music music))
            return music;

        return null;
    }

}
