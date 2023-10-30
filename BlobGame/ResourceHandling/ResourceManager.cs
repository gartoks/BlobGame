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
    private static ConcurrentDictionary<string, (Font? font, FontResource resource)> Fonts { get; }
    /// <summary>
    /// Texture resources.
    /// </summary>
    private static ConcurrentDictionary<string, (Texture? texture, TextureResource resource)> Textures { get; }
    /// <summary>
    /// Sound resources.
    /// </summary>
    private static ConcurrentDictionary<string, (Sound? sound, SoundResource resource)> Sounds { get; }
    /// <summary>
    /// Sound resources.
    /// </summary>
    private static ConcurrentDictionary<string, (Music? music, MusicResource resource)> Music { get; }
    /// <summary>
    /// Text resources.
    /// </summary>
    private static ConcurrentDictionary<string, (string? text, TextResource resource)> Texts { get; }

    /// <summary>
    /// Default raylib font.
    /// </summary>
    private static Font _FallbackFont { get; set; }
    /// <summary>
    /// Default fallback font resource.
    /// </summary>
    public static FontResource FallbackFont { get; private set; }

    /// <summary>
    /// Default raylib texture.
    /// </summary>
    private static Texture _FallbackTexture { get; set; }
    /// <summary>
    /// Default fallback texture resource.
    /// </summary>
    public static TextureResource FallbackTexture { get; private set; }

    /// <summary>
    /// Default raylib sound.
    /// </summary>
    private static Sound _FallbackSound { get; set; }
    /// <summary>
    /// Default fallback sound resource.
    /// </summary>
    public static SoundResource FallbackSound { get; private set; }

    /// <summary>
    /// Default raylib music.
    /// </summary>
    private static Music _FallbackMusic { get; set; }
    /// <summary>
    /// Default fallback music resource.
    /// </summary>
    public static MusicResource FallbackMusic { get; private set; }

    /// <summary>
    /// Default text.
    /// </summary>
    private static string _FallbackText { get; set; }
    /// <summary>
    /// Default fallback text resource.
    /// </summary>
    public static TextResource FallbackText { get; private set; }

    /// <summary>
    /// Base theme.
    /// </summary>
    private static Theme _DefaultTheme { get; }
    /// <summary>
    /// Additional theme.
    /// </summary>
    public static Theme MainTheme { get; private set; }

    /// <summary>
    /// Static constructor to initialize the resource loading queue and other required properties.
    /// </summary>
    static ResourceManager() {
        ResourceLoadingQueue = new BlockingCollection<(string key, Type type)>();

        Fonts = new();
        Textures = new();
        Sounds = new();
        Music = new();
        Texts = new();

        _DefaultTheme = new Theme(Files.GetResourceFilePath("MelbaToast.theme"));
        MainTheme = _DefaultTheme;
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
        _DefaultTheme.Load();

        _FallbackFont = Raylib.GetFontDefault();
        Image image = Raylib.GenImageColor(1, 1, Raylib.BLANK);
        _FallbackTexture = Raylib.LoadTextureFromImage(image);
        _FallbackSound = new Sound();
        _FallbackMusic = new Music();
        _FallbackText = string.Empty;

        FallbackFont = new FontResource("fallback", _FallbackFont, TryGetFont);
        FallbackTexture = new TextureResource("fallback", _FallbackTexture, TryGetTexture);
        FallbackSound = new SoundResource("fallback", _FallbackSound, TryGetSound);
        FallbackMusic = new MusicResource("fallback", _FallbackMusic, TryGetMusic);
        FallbackText = new TextResource("fallback", _FallbackText, TryGetText);
    }

    internal static void Unload() {
        _DefaultTheme?.Unload();
        _DefaultTheme?.Dispose();

        MainTheme?.Unload();
        MainTheme?.Dispose();
    }

    /// <summary>
    /// Clears the resource cache.
    /// </summary>
    private static void UnloadEverything() {
        while (ResourceLoadingQueue.TryTake(out _)) ;
        Debug.WriteLine("Unloading fonts.");
        foreach (string key in Fonts.Keys.ToList()) {
            UnloadFont(key);
            LoadFont(key);
        }

        Debug.WriteLine("Unloading textures.");
        foreach (string key in Textures.Keys.ToList()) {
            UnloadTexture(key);
            LoadTexture(key);
        }

        Debug.WriteLine("Unloading sounds.");
        foreach (string key in Sounds.Keys.ToList()) {
            UnloadSound(key);
            LoadSound(key);
        }

        Debug.WriteLine("Unloading music.");
        foreach (string key in Music.Keys.ToList()) {
            UnloadMusic(key);
            LoadMusic(key);
        }

        Debug.WriteLine("Unloading texts.");
        foreach (string key in Texts.Keys.ToList()) {
            UnloadText(key);
            LoadText(key);
        }
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
            if (!Fonts.TryGetValue(key, out (Font? font, FontResource resource) val)) {
                Debug.WriteLine($"Unable to load font '{key}'.");
                return;
            }

            Font? font = MainTheme.LoadFont(key) ?? _DefaultTheme.LoadFont(key);
            if (font == null) {
                Debug.WriteLine($"The default theme doesn't contain a font for {key}");
                return;
            }

            Fonts[key] = (font.Value, val.resource);
        } else if (type == typeof(Texture)) {
            if (!Textures.TryGetValue(key, out (Texture? texture, TextureResource resource) val)) {
                Debug.WriteLine($"Unable to load texture '{key}'.");
                return;
            }

            Texture? texture = MainTheme.LoadTexture(key) ?? _DefaultTheme.LoadTexture(key);
            if (texture == null) {
                Debug.WriteLine($"The default theme doesn't contain a texture for {key}");
                return;
            }

            Textures[key] = (texture.Value, val.resource);
        } else if (type == typeof(Sound)) {
            if (!Sounds.TryGetValue(key, out (Sound? sound, SoundResource resource) val)) {
                Debug.WriteLine($"Unable to load sound '{key}'.");
                return;
            }

            Sound? sound = MainTheme.LoadSound(key) ?? _DefaultTheme.LoadSound(key);
            if (sound == null) {
                Debug.WriteLine($"The default theme doesn't contain a sound for {key}");
                return;
            }

            Sounds[key] = (sound.Value, val.resource);
        } else if (type == typeof(Music)) {
            if (!Music.TryGetValue(key, out (Music? music, MusicResource resource) val)) {
                Debug.WriteLine($"Unable to load music '{key}'.");
                return;
            }

            Music? music = MainTheme.LoadMusic(key) ?? _DefaultTheme.LoadMusic(key);
            if (music == null) {
                Debug.WriteLine($"The default theme doesn't contain a music for {key}");
                return;
            }

            Music[key] = (music.Value, val.resource);
        } else if (type == typeof(string)) {
            if (!Texts.TryGetValue(key, out (string? text, TextResource resource) val)) {
                Debug.WriteLine($"Unable to load text '{key}'.");
                return;
            }

            string? text = MainTheme.LoadText(key) ?? _DefaultTheme.LoadText(key);
            if (text == null) {
                Debug.WriteLine($"The default theme doesn't contain a text for {key}");
                return;
            }

            Texts[key] = (text!, val.resource);
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

        if (MainTheme != _DefaultTheme) {
            MainTheme.Unload();
            MainTheme.Dispose();
        }

        MainTheme = new Theme(filename);
        MainTheme.Load();

        // force everything to reload
        UnloadEverything();
    }

    /// <summary>
    /// Tries to get a color to the given key.
    /// </summary>
    /// <param name="key"></param>
    public static ColorResource GetColor(string key) {
        return new ColorResource(key, GetColorInternal);
    }

    /// <summary>
    /// Tries to get a raylib color from the given key. Retrieves it from the theme if needed.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    private static Color GetColorInternal(string key) {
        return MainTheme.GetColor(key) ?? _DefaultTheme.GetColor(key) ?? Raylib.RED;
    }

    /// <summary>
    /// Gets a font resource from the given key. Queues it for loading if needed.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public static FontResource GetFont(string key) {
        LoadFont(key);
        return Fonts[key].resource;
    }

    /// <summary>
    /// Queues a font for loading from the given key.
    /// </summary>
    /// <param name="key"></param>
    private static void LoadFont(string key) {
        if (ResourceLoadingQueue.Any(r => r.key == key)) {
            return;
        }
        if (Fonts.Any(pair => pair.Key == key && pair.Value.resource.IsLoaded())) {
            return;
        }

        if (!Fonts.ContainsKey(key)) {
            Fonts[key] = (null, new FontResource(key, _FallbackFont, TryGetFont));
            ResourceLoadingQueue.Add((key, typeof(Font)));
        } else if (Fonts.Any(pair => pair.Key == key && !pair.Value.resource.IsLoaded())) {
            ResourceLoadingQueue.Add((key, typeof(Font)));
        }
    }

    /// <summary>
    /// Tries to get a raylib font from the given key. Returns null if it doesn't exist.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    private static Font? TryGetFont(string key) {
        if (Fonts.TryGetValue(key, out (Font? font, FontResource resource) font))
            return font.font;

        return null;
    }

    /// <summary>
    /// Unloads the given font.
    /// </summary>
    /// <param name="key"></param>
    private static void UnloadFont(string key) {
        FontResource resource = Fonts[key].resource;
        if (resource.Resource.texture.id != 0 && resource.Resource.texture.id != FallbackFont.Resource.texture.id)
            Raylib.UnloadFont(resource.Resource);
        resource.Unload();
        Fonts[key] = (null, resource);
    }

    /// <summary>
    /// Gets a texture resource from the given key. Queues it for loading if needed.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public static TextureResource GetTexture(string key) {
        LoadTexture(key);
        return Textures[key].resource;
    }

    /// <summary>
    /// Queues a texture for loading from the given key.
    /// </summary>
    /// <param name="key"></param>
    private static void LoadTexture(string key) {
        if (!Textures.Any(pair => pair.Key == key && pair.Value.resource.IsLoaded()) && !ResourceLoadingQueue.Any(r => r.key == key)) {
            Textures[key] = (null, new TextureResource(key, _FallbackTexture, TryGetTexture));
            ResourceLoadingQueue.Add((key, typeof(Texture)));
        }
    }

    /// <summary>
    /// Tries to get a raylib texture from the given key. Returns null if it doesn't exist.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    private static Texture? TryGetTexture(string key) {
        if (Textures.TryGetValue(key, out (Texture? texture, TextureResource resource) texture))
            return texture.texture;

        return null;
    }

    /// <summary>
    /// Unloads the given texture.
    /// </summary>
    /// <param name="key"></param>
    private static void UnloadTexture(string key) {
        TextureResource resource = Textures[key].resource;
        if (resource.Resource.id != 0 && resource.Resource.id != FallbackTexture.Resource.id)
            Raylib.UnloadTexture(resource.Resource);
        resource.Unload();
        Textures[key] = (null, resource);
    }

    /// <summary>
    /// Gets a sound resource from the given key. Queues it for loading if needed.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public static SoundResource GetSound(string key) {
        LoadSound(key);
        return Sounds[key].resource;
    }

    /// <summary>
    /// Queues a sound for loading from the given key.
    /// </summary>
    /// <param name="key"></param>
    private static void LoadSound(string key) {
        if (!Sounds.Any(pair => pair.Key == key && pair.Value.resource.IsLoaded()) && !ResourceLoadingQueue.Any(r => r.key == key)) {
            Sounds[key] = (null, new SoundResource(key, _FallbackSound, TryGetSound));
            ResourceLoadingQueue.Add((key, typeof(Sound)));
        }
    }

    /// <summary>
    /// Tries to get a raylib sound from the given key. Returns null if it doesn't exist.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    private static Sound? TryGetSound(string key) {
        if (Sounds.TryGetValue(key, out (Sound? sound, SoundResource resource) sound))
            return sound.sound;

        return null;
    }

    /// <summary>
    /// Unloads the given sound.
    /// </summary>
    /// <param name="key"></param>
    private static void UnloadSound(string key) {
        SoundResource resource = Sounds[key].resource;
        Raylib.UnloadSound(resource.Resource);
        resource.Unload();
        Sounds[key] = (null, resource);
    }

    /// <summary>
    /// Gets a Music resource from the given key. Queues it for loading if needed.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public static MusicResource GetMusic(string key) {
        LoadMusic(key);
        return Music[key].resource;
    }

    /// <summary>
    /// Queues a Music for loading from the given key.
    /// </summary>
    /// <param name="key"></param>
    private static void LoadMusic(string key) {
        if (!Music.Any(pair => pair.Key == key && pair.Value.resource.IsLoaded()) && !ResourceLoadingQueue.Any(r => r.key == key)) {
            Music[key] = (null, new MusicResource(key, _FallbackMusic, TryGetMusic));
            ResourceLoadingQueue.Add((key, typeof(Music)));
        }
    }

    /// <summary>
    /// Tries to get a raylib Music from the given key. Returns null if it doesn't exist.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    private static Music? TryGetMusic(string key) {
        if (Music.TryGetValue(key, out (Music? music, MusicResource resource) music))
            return music.music;

        return null;
    }

    /// <summary>
    /// Unloads the given sound.
    /// </summary>
    /// <param name="key"></param>
    private static void UnloadMusic(string key) {
        MusicResource resource = Music[key].resource;
        Raylib.UnloadMusicStream(resource.Resource);
        resource.Unload();
        Music[key] = (null, resource);
    }

    /// <summary>
    /// Gets a text resource from the given key. Queues it for loading if needed.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    public static TextResource GetText(string key) {
        LoadText(key);
        return Texts[key].resource;
    }

    /// <summary>
    /// Queues a text for loading from the given key.
    /// </summary>
    /// <param name="key"></param>
    private static void LoadText(string key) {
        if (!Texts.Any(pair => pair.Key == key && pair.Value.resource.IsLoaded()) && !ResourceLoadingQueue.Any(r => r.key == key)) {
            Texts[key] = (null, new TextResource(key, _FallbackText, TryGetText));
            ResourceLoadingQueue.Add((key, typeof(string)));
        }
    }

    /// <summary>
    /// Tries to get a string from the given key. Returns null if it doesn't exist.
    /// </summary>
    /// <param name="key"></param>
    /// <returns></returns>
    private static string? TryGetText(string key) {
        if (Texts.TryGetValue(key, out (string? text, TextResource resource) text))
            return text.text;

        return null;
    }

    /// <summary>
    /// Unloads the given text.
    /// </summary>
    /// <param name="key"></param>
    private static void UnloadText(string key) {
        TextResource resource = Texts[key].resource;
        resource.Unload();
        Texts[key] = (null, resource);
    }
}
