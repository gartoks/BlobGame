using Raylib_CsLo;
using System.Collections.Concurrent;
using System.Numerics;

namespace BlobGame.ResourceHandling.Resources;
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

    internal void Draw(string text, float fontSize, ColorResource tint, Vector2 position, Vector2? pivot = null, float rotation = 0) {
        Draw(text, fontSize, tint.Resource, position, pivot, rotation);
    }
    internal void Draw(string text, float fontSize, Color tint, Vector2 position, Vector2? pivot = null, float rotation = 0) {
        if (pivot == null)
            pivot = Vector2.Zero;

        Vector2 textSize = Raylib.MeasureTextEx(Resource, text, fontSize, fontSize / 16f);

        Raylib.DrawTextPro(
            Resource,
            text,
            position,
            textSize * pivot.Value,
            rotation,
            fontSize,
            fontSize / 16f,
            tint);
    }
}

internal sealed class FontResourceLoader : ResourceLoader<Font, FontResource> {
    public FontResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue)
        : base(resourceLoadingQueue) {
    }

    protected override bool ResourceExistsInternal(string key) {
        return ResourceManager.MainTheme.DoesFontExist(key);
    }

    protected override Font LoadResourceInternal(string key) {
        Font? res = ResourceManager.MainTheme.LoadFont(key) ?? ResourceManager.DefaultTheme.LoadFont(key);
        return res ?? Fallback.Resource;
    }

    protected override void UnloadResourceInternal(FontResource resource) {
        if (resource.Resource.texture.id != 0 && resource.Resource.texture.id != Fallback.Resource.texture.id)
            Raylib.UnloadFont(resource.Resource);
    }
}