using Raylib_CsLo;

namespace BlobGame.ResourceHandling;
public sealed class FontResource {
    public string Key { get; }

    private Font Fallback { get; }
    private Func<string, Font?> ResourceRetriever { get; }

    private Font? _Resource { get; set; }
    public Font Resource {
        get {
            if (_Resource == null)
                _Resource = ResourceRetriever(Key);

            return _Resource ?? Fallback;
        }
    }

    internal FontResource(string key, Font fallback, Func<string, Font?> resourceRetriever) {
        Key = key;

        Fallback = fallback;
        ResourceRetriever = resourceRetriever;
        _Resource = resourceRetriever(key);
    }
}
