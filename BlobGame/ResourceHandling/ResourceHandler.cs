using BlobGame.App;
using Raylib_CsLo;
using System.Collections.Concurrent;
using System.Diagnostics;

namespace BlobGame.ResourceHandling;
internal static class ResourceHandler {
    private const int RESOURCE_LOADING_TIMEOUT = 0;
    private static BlockingCollection<(string key, Type type)> ResourceLoadingQueue { get; }

    private static ConcurrentDictionary<string, Font> Fonts { get; }
    private static ConcurrentDictionary<string, Texture> Textures { get; }
    private static ConcurrentDictionary<string, Sound> Sounds { get; }

    private static Texture _DefaultTexture { get; set; }
    public static FontResource DefaultFont { get; private set; }

    private static Font _DefaultFont { get; set; }
    public static TextureResource DefaultTexture { get; private set; }

    private static Sound _DefaultSound { get; set; }
    public static SoundResource DefaultSound { get; private set; }

    static ResourceHandler() {
        ResourceLoadingQueue = new BlockingCollection<(string key, Type type)>();

        Fonts = new ConcurrentDictionary<string, Font>();
        Textures = new ConcurrentDictionary<string, Texture>();
        Sounds = new ConcurrentDictionary<string, Sound>();
    }

    internal static void Initialize() {
    }

    internal static void Load() {
        _DefaultFont = Raylib.GetFontDefault();
        Image image = Raylib.GenImageColor(1, 1, Raylib.BLANK);
        _DefaultTexture = Raylib.LoadTextureFromImage(image);

        DefaultFont = new FontResource("default", _DefaultFont, TryGetFont);
        DefaultTexture = new TextureResource("default", _DefaultTexture, TryGetTexture);
        DefaultSound = new SoundResource("default", _DefaultSound, TryGetSound);
    }

    internal static void Update() {
        while (ResourceLoadingQueue.TryTake(out (string key, Type type) resource, RESOURCE_LOADING_TIMEOUT)) {
            LoadResource(resource.key, resource.type);
        }
    }

    private static void LoadResource(string key, Type type) {
        if (type == typeof(Font)) {
            if (Fonts.ContainsKey(key))
                return;

            string path = Files.GetResourceFilePath("Fonts", $"{key}.ttf");

            Font font = Raylib.LoadFont(path);

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
        } else {
            Debug.WriteLine($"Resource type {type} is not supported");
        }
    }

    public static void LoadFont(string key) {
        if (!Fonts.ContainsKey(key) && !ResourceLoadingQueue.Any(r => r.key == key))
            ResourceLoadingQueue.Add((key, typeof(Font)));
    }

    public static FontResource GetFont(string key) {
        LoadFont(key);
        return new FontResource(key, _DefaultFont, TryGetFont);
    }

    private static Font? TryGetFont(string key) {
        if (Fonts.TryGetValue(key, out Font font))
            return font;

        return null;
    }


    public static void LoadTexture(string key) {
        if (!Textures.ContainsKey(key))
            ResourceLoadingQueue.Add((key, typeof(Texture)));
    }

    public static TextureResource GetTexture(string key) {
        LoadTexture(key);
        return new TextureResource(key, _DefaultTexture, TryGetTexture);
    }

    private static Texture? TryGetTexture(string key) {
        if (Textures.TryGetValue(key, out Texture texture))
            return texture;

        return null;
    }

    public static void LoadSound(string key) {
        if (!Sounds.ContainsKey(key))
            ResourceLoadingQueue.Add((key, typeof(Sound)));
    }

    public static SoundResource GetSound(string key) {
        LoadSound(key);
        return new SoundResource(key, _DefaultSound, TryGetSound);
    }

    private static Sound? TryGetSound(string key) {
        if (Sounds.TryGetValue(key, out Sound sound))
            return sound;

        return null;
    }

}
