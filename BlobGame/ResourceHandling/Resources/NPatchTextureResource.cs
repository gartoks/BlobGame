﻿using Raylib_CsLo;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;

namespace BlobGame.ResourceHandling.Resources;

internal record NPatchTexture(Texture Texture, int left, int right, int top, int bottom);

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

    internal void Draw(Rectangle bounds, Vector2? pivot = null, Color? tint = null) {
        if (pivot == null)
            pivot = Vector2.Zero;

        float tw = Resource.Texture.width;
        float th = Resource.Texture.height;
        float bw = bounds.width;
        float bh = bounds.height;
        float r = Resource.Texture.width - Resource.right;
        float b = Resource.Texture.height - Resource.bottom;

        float centerW = Math.Max(0, bounds.width - r - Resource.left);
        float centerH = Math.Max(0, bounds.height - b - Resource.top);
        float wScale = Math.Min(1, bw / (Resource.left + r));
        float hScale = Math.Min(1, bh / (Resource.top + b));

        void Draw(float xT, float yT, float wT, float hT, float xB, float yB, float wB, float hB) {
            Raylib.DrawTexturePro(
                    Resource.Texture,
                    new Rectangle(xT, yT, wT, hT),
                    new Rectangle(bounds.x + xB, bounds.y + yB, wB, hB),
                    Vector2.Zero,   // TODO
                    0,  // TODO
                    tint != null ? tint.Value : Raylib.WHITE);
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

        /*float x0 = 0;
        float y0 = 0;
        float w0 = Resource.left / (float)Resource.Texture.width;
        float h0 = Resource.top / (float)Resource.Texture.height;
        float x1 = w0;
        float y1 = h0;
        float w1 = Resource.right / (float)Resource.Texture.width - w0;
        float h1 = Resource.bottom / (float)Resource.Texture.height - h0;
        float x2 = w0 + w1;
        float y2 = h0 + h1;
        float w2 = 1 - w0 - w1;
        float h2 = 1 - h0 - h1;

        float[] x = new float[] { x0, x1, x2 };
        float[] y = new float[] { y0, y1, y2 };
        float[] w = new float[] { w0, w1, w2 };
        float[] h = new float[] { h0, h1, h2 };

        for (int yi = 0; yi <= 2; yi++) {
            for (int xi = 0; xi <= 2; xi++) {
                Raylib.DrawTexturePro(
                    Resource.Texture,
                    new Rectangle(x[xi] * tw, y[yi] * th, w[xi] * tw, h[yi] * th),
                    new Rectangle(bounds.x + x[xi] * bw, bounds.y + y[yi] * bh, w[xi] * bw, h[yi] * bh),
                    Vector2.Zero,   // TODO
                    0,  // TODO
                    tint != null ? tint.Value : Raylib.WHITE);
            }
        }*/
    }
}

internal sealed class NPatchTextureResourceLoader : ResourceLoader<NPatchTexture, NPatchTextureResource> {
    public NPatchTextureResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue)
        : base(resourceLoadingQueue) {
    }

    protected override NPatchTexture LoadResourceInternal(string key) {
        NPatchTexture? res = ResourceManager.MainTheme.LoadNPatchTexture(key) ?? ResourceManager.DefaultTheme.LoadNPatchTexture(key);
        return res ?? Fallback.Resource;
    }

    protected override void UnloadResourceInternal(NPatchTextureResource resource) {
        if (resource.Resource.Texture.id != 0 && resource.Resource.Texture.id != Fallback.Resource.Texture.id)
            Raylib.UnloadTexture(resource.Resource.Texture);
    }
}
