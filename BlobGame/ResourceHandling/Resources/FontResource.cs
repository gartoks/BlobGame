using SimpleGL.Util;
using System.Collections.Concurrent;

namespace BlobGame.ResourceHandling.Resources;
/*
/// <summary>
/// Game resource for fonts.
/// </summary>
internal sealed class FontResource : GameResource<Font> {
    /// <summary>
    /// Constructor for a new font resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="fallback"></param>
    /// <param name="resourceRetriever"></param>
    internal FontResource(string key, Font fallback, ResourceRetrieverDelegate resourceRetriever)
        : base(key, fallback, resourceRetriever) {

    }

    internal void Draw(string text, float fontSize, ColorResource tint, Vector2 position, *//*Vector2? pivot = null, *//*float rotation = 0) {
        Draw(text, fontSize, tint.Resource, position, *//*pivot, *//*rotation);
    }
    internal void Draw(string text, float fontSize, Color4 tint, Vector2 position, *//*Vector2? pivot = null, *//*float rotation = 0) {
        *//*if (pivot == null)
            pivot = Vector2.Zero;
*//*
        Raylib.DrawTextPro(
            Resource,
            text,
            position,
            new Vector2(),
            rotation,
            fontSize,
            fontSize / 16f,
            tint);
    }
}*/

internal sealed class FontResourceLoader : ResourceLoader<FontFamilyData> {
    public FontResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue)
        : base(resourceLoadingQueue) {
    }

    protected override FontFamilyData LoadResourceInternal(string key) {
        FontFamilyData? res = ResourceManager.MainTheme.LoadFont(key) ?? ResourceManager.DefaultTheme.LoadFont(key);
        return res ?? throw new ArgumentException($"Font resource '{key}' does not exist in theme.");
    }

    protected override void UnloadResourceInternal(string key, FontFamilyData resource) {
    }
}