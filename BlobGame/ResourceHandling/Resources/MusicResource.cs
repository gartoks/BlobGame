using BlobGame.Audio;
using System.Collections.Concurrent;

namespace BlobGame.ResourceHandling.Resources;
/*
/// <summary>
/// Game resource for music.
/// </summary>
internal sealed class MusicResource : GameResource<Music> {
    /// <summary>
    /// Constructor for a new music resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="fallback"></param>
    /// <param name="resourceRetriever"></param>
    internal MusicResource(string key, Music fallback, ResourceRetrieverDelegate resourceRetriever)
        : base(key, fallback, resourceRetriever) {
    }
}
*/
internal sealed class MusicResourceLoader : ResourceLoader<Music> {
    public MusicResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue)
        : base(resourceLoadingQueue) {
    }

    protected override Music LoadResourceInternal(string key) {
        Music? res = ResourceManager.MainTheme.LoadMusic(key) ?? ResourceManager.DefaultTheme.LoadMusic(key);
        return res ?? throw new ArgumentException($"Music resource '{key}' does not exist in theme.");
    }

    protected override void UnloadResourceInternal(string key, Music resource) {
        resource.Dispose();
    }
}
