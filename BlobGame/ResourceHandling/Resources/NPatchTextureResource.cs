using SimpleGL.Graphics.Textures;
using System.Collections.Concurrent;

namespace BlobGame.ResourceHandling.Resources;

/*
/// <summary>
/// Game resource for npatch textures.
/// </summary>
internal sealed class NPatchTextureResource : GameResource<NPatchTexture> {
    /// <summary>
    /// Constructor for a new npatch texture resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="fallback"></param>
    /// <param name="resourceRetriever"></param>
    internal NPatchTextureResource(string key, NPatchTexture fallback, ResourceRetrieverDelegate resourceRetriever)
        : base(key, fallback, resourceRetriever) {
    }

    internal void Draw(Box2 bounds, Vector2? pivot = null, Color4? tint = null) {
        if (pivot == null)
            pivot = Vector2.Zero;

        float tw = Resource.Texture.Width;
        float th = Resource.Texture.Height;
        float bw = bounds.Width();
        float bh = bounds.Height();
        float r = Resource.Texture.Width - Resource.right;
        float b = Resource.Texture.Height - Resource.bottom;

        float centerW = Math.Max(0, bounds.Width() - r - Resource.left);
        float centerH = Math.Max(0, bounds.Height() - b - Resource.top);
        float wScale = Math.Min(1, bw / (Resource.left + r));
        float hScale = Math.Min(1, bh / (Resource.top + b));

        void Draw(float xT, float yT, float wT, float hT, float xB, float yB, float wB, float hB) {
            Raylib.DrawTexturePro(
                    Resource.Texture,
                    new Rectangle(xT, yT, wT, hT),
                    new Rectangle(bounds.x + xB, bounds.y + yB, wB, hB),
                    Vector2.Zero,   // TODO
                    0,  // TODO
                    tint != null ? tint.Value : Color4.White);
        }

        Debug.WriteLine("");
        // Top left
        Draw(
            0, 0,
            Resource.left, Resource.top,
            0, 0,
            Resource.left * wScale, Resource.top * hScale);
        // Top right
        Draw(
            Resource.right, 0,
            r, Resource.top,
            bw - r * wScale, 0,
            r * wScale, Resource.top * hScale);
        // Bottom left
        Draw(
            0, Resource.bottom,
            Resource.left, b,
            0, bh - b * hScale,
            Resource.left * wScale, b * hScale);
        // Bottom right
        Draw(
            Resource.right, Resource.bottom,
            r, b,
            bw - r * wScale, bh - b * hScale,
            r * wScale, b * hScale);
        if (centerW > 0) {
            // Top
            Draw(
                Resource.left, 0,
                Resource.right - Resource.left, Resource.top,
                Resource.left * wScale, 0,
                centerW, Resource.top * hScale);
            // Bottom
            Draw(
                Resource.left, Resource.bottom,
                Resource.right - Resource.left, b,
                Resource.left * wScale, bh - b * hScale,
                centerW, b * hScale);
        }
        if (centerH > 0) {
            // Left
            Draw(
                0, Resource.top,
                Resource.left, Resource.bottom - Resource.top,
                0, Resource.top * hScale,
                Resource.left * wScale, centerH);
            // Right
            Draw(
                Resource.right, Resource.top,
                Resource.left, Resource.bottom - Resource.top,
                bw - r * wScale, Resource.top * hScale,
                Resource.left * wScale, centerH);
        }
        if (centerW > 0 && centerH > 0) {
            // Center
            Draw(
                Resource.left, Resource.top,
                Resource.right - Resource.left, Resource.bottom - Resource.top,
                Resource.left * wScale, Resource.top * hScale,
                centerW, centerH);
        }
    }
}*/

internal sealed class NPatchTextureResourceLoader : ResourceLoader<NPatchTexture> {
    public NPatchTextureResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue)
        : base(resourceLoadingQueue) {
    }

    protected override NPatchTexture LoadResourceInternal(string key) {
        NPatchTexture? res = ResourceManager.MainTheme.LoadNPatchTexture(key) ?? ResourceManager.DefaultTheme.LoadNPatchTexture(key);
        return res ?? throw new ArgumentException($"NPatchTexture resource '{key}' does not exist in theme.");
    }

    protected override void UnloadResourceInternal(string key, NPatchTexture resource) {
        resource.Texture.Dispose();
    }
}
