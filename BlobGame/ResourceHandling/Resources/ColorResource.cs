using OpenTK.Mathematics;
using System.Collections.Concurrent;

namespace BlobGame.ResourceHandling.Resources;
/*
/// <summary>
/// Game resource for colors.
/// </summary>
internal sealed class ColorResource : GameResource<Color4> {
    public static ColorResource WHITE { get; } = new ColorResource("white", Color4.White, _ => Color4.White);
    /// <summary>
    /// Constructor for a new color resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="resourceRetriever"></param>
    internal ColorResource(string key, Color4 fallback, ResourceRetrieverDelegate resourceRetriever)
        : base(key, fallback, resourceRetriever) {
    }
}*/

internal sealed class ColorResourceLoader : ResourceLoader<Color4> {
    public ColorResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue)
   : base(resourceLoadingQueue) {
    }

    protected override Color4 LoadResourceInternal(string key) {
        Color4? res = ResourceManager.MainTheme.GetColor(key) ?? ResourceManager.DefaultTheme.GetColor(key);
        return res ?? throw new ArgumentException($"Color resource '{key}' does not exist in theme.");
    }

    protected override void UnloadResourceInternal(string key, Color4 resource) {
        throw new NotImplementedException();
    }
}