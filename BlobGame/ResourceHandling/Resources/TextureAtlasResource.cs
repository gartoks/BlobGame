using Raylib_CsLo;
using System.Collections.Concurrent;
using System.Numerics;

namespace BlobGame.ResourceHandling.Resources;

internal record TextureAtlas(Texture Texture, IReadOnlyDictionary<string, (int x, int y, int w, int h)> SubTextures);

/// <summary>
/// Game resource for npatch textures.
/// </summary>
internal sealed class TextureAtlasResource : GameResource<TextureAtlas> {
    /// <summary>
    /// Constructor for a new npatch texture resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="fallback"></param>
    /// <param name="resourceRetriever"></param>
    internal TextureAtlasResource(string key, TextureAtlas fallback, ResourceRetrieverDelegate resourceRetriever)
        : base(key, fallback, resourceRetriever) {
    }

    internal void DrawAsBitmapFont(string text, float spacing, float height, Vector2 position, Vector2? pivot = null, Color? tint = null) {
        if (text.Length == 0)
            return;

        if (pivot == null)
            pivot = Vector2.Zero;

        List<KeyValuePair<char, (int x, int y, int w, int h)>> subTextures = new();

        for (int i = 0; i < text.Length; i++) {
            char c = text[i];

            if (Resource.SubTextures.TryGetValue(c.ToString(), out (int x, int y, int w, int h) subTexture))
                subTextures.Add(new KeyValuePair<char, (int x, int y, int w, int h)>(c, subTexture));
        }

        float maxH = subTextures.Select(t => t.Value.h).Max();
        float heightScale = height / maxH;

        float totalWidth = subTextures.Select(t => t.Value.w).Sum() * heightScale + (subTextures.Count - 1) * spacing;
        float x = position.X - totalWidth * pivot.Value.X;
        float y = position.Y - height * pivot.Value.Y;

        for (int i = 0; i < subTextures.Count; i++) {
            KeyValuePair<char, (int x, int y, int w, int h)> subTexture = subTextures[i];

            float tx = subTexture.Value.x;
            float ty = subTexture.Value.y;
            float tw = subTexture.Value.w;
            float th = subTexture.Value.h;

            float w = heightScale * tw;
            float h = height;

            Raylib.DrawTexturePro(
                    Resource.Texture,
                    new Rectangle(tx, ty, tw, th),
                    new Rectangle(x, y, w, h),
                    Vector2.Zero,
                    0,
                    tint != null ? tint.Value : Raylib.WHITE);

            x += w + spacing;
        }
    }
}

internal sealed class TextureAtlasResourceLoader : ResourceLoader<TextureAtlas, TextureAtlasResource> {
    public TextureAtlasResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue)
        : base(resourceLoadingQueue) {
    }

    protected override TextureAtlas LoadResourceInternal(string key) {
        TextureAtlas? res = ResourceManager.MainTheme.LoadTextureAtlas(key) ?? ResourceManager.DefaultTheme.LoadTextureAtlas(key);
        return res ?? Fallback.Resource;
    }

    protected override void UnloadResourceInternal(TextureAtlasResource resource) {
        if (resource.Resource.Texture.id != 0 && resource.Resource.Texture.id != Fallback.Resource.Texture.id)
            Raylib.UnloadTexture(resource.Resource.Texture);
    }
}
