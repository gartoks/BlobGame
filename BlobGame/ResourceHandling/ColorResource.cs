using Raylib_CsLo;

namespace BlobGame.ResourceHandling;
/// <summary>
/// Game resource for colors.
/// </summary>
internal readonly struct ColorResource {
    public static ColorResource WHITE { get; } = new ColorResource("white", _ => Raylib.WHITE);

    /// <summary>
    /// The key of the resource.
    /// </summary>
    public string Key { get; }

    /// <summary>
    /// Function to retrieve the resource from the resource manager. Used to check if the resource has been loaded.
    /// </summary>
    private Func<string, Color> ResourceRetriever { get; }

    /// <summary>
    /// The raylib resource. Returns the fallback if the resource has not been loaded (yet).
    /// </summary>
    public Color Resource {
        get {
            return ResourceRetriever(Key);
        }
    }

    /// <summary>
    /// Constructor for a new color resource.
    /// </summary>
    /// <param name="key"></param>
    /// <param name="resourceRetriever"></param>
    internal ColorResource(string key, Func<string, Color> resourceRetriever) {
        Key = key;
        ResourceRetriever = resourceRetriever;
    }
}
