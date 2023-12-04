using BlobGame.Audio;
using System.Collections.Concurrent;

namespace BlobGame.ResourceHandling.Resources;
/*
/// <summary>
/// Game resource for sounds.
/// </summary>
internal sealed class SoundResource : GameResource<Sound> {
    /// <summary>
    /// Constructor for a new sound resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="fallback"></param>
    /// <param name="resourceRetriever"></param>
    internal SoundResource(string key, Sound fallback, ResourceRetrieverDelegate resourceRetriever)
        : base(key, fallback, resourceRetriever) {
    }
}
*/
internal sealed class SoundResourceLoader : ResourceLoader<Sound> {
    public SoundResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue)
        : base(resourceLoadingQueue) {
    }

    protected override Sound LoadResourceInternal(string key) {
        Sound? res = ResourceManager.MainTheme.LoadSound(key) ?? ResourceManager.DefaultTheme.LoadSound(key);
        return res ?? throw new ArgumentException($"Sound resource '{key}' does not exist in theme.");
    }

    protected override void UnloadResourceInternal(string key, Sound resource) {
        resource.Dispose();
    }
}