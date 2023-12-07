using System.Collections.Concurrent;

namespace BlobGame.ResourceHandling.Resources;
/// <summary>
/// Game resource for text.
/// </summary>
internal sealed class TextResource : GameResource<IReadOnlyDictionary<string, string>> {
    /// <summary>
    /// Constructor for a new text resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="resourceRetriever"></param>
    internal TextResource(string key, IReadOnlyDictionary<string, string> fallback, ResourceRetrieverDelegate resourceRetriever)
        : base(key, fallback, resourceRetriever) {
    }
}

internal sealed class TextResourceLoader : ResourceLoader<IReadOnlyDictionary<string, string>, TextResource> {
    public TextResourceLoader(BlockingCollection<(string key, Type type)> resourceLoadingQueue)
        : base(resourceLoadingQueue) {
    }

    protected override bool ResourceExistsInternal(string key) {
        return ResourceManager.MainTheme.DoesTextExist(key);
    }

    protected override IReadOnlyDictionary<string, string> LoadResourceInternal(string key) {
        IReadOnlyDictionary<string, string>? res = ResourceManager.MainTheme.LoadText(key) ?? ResourceManager.DefaultTheme.LoadText(key);
        return res ?? Fallback.Resource;
    }

    protected override void UnloadResourceInternal(TextResource resource) {
    }
}