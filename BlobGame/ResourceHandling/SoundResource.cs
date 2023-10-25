using Raylib_CsLo;

namespace BlobGame.ResourceHandling;

public class SoundResource {
    public string Key { get; }

    private Sound Fallback { get; }
    private Func<string, Sound?> ResourceRetriever { get; }

    private Sound? _Reource { get; set; }
    public Sound Resource {
        get {
            if (_Reource == null)
                _Reource = ResourceRetriever(Key);

            return _Reource ?? Fallback;
        }
    }

    internal SoundResource(string key, Sound fallback, Func<string, Sound?> resourceRetriever) {
        Key = key;

        ResourceRetriever = resourceRetriever;
        Fallback = fallback;
        _Reource = resourceRetriever(key);
    }

    /*internal void DrawScreen(RectangleF destinationRect, Color? tint = null, Vector2? origin = null, float rotation = 0) {
        if (tint == null)
            tint = Raylib.WHITE;

        if (origin == null)
            origin = Vector2.Zero;

        Rectangle sourceRect = new Rectangle(0, 0, Resource.width, Resource.height);
        Raylib.DrawTexturePro(Resource, sourceRect, destinationRect.ToRaylib(), origin.Value, RayMath.RAD2DEG * rotation, tint.Value);
    }*/
}
