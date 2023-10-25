using Raylib_CsLo;
using System.Drawing;
using System.Numerics;
using Color = Raylib_CsLo.Color;
using Rectangle = Raylib_CsLo.Rectangle;

namespace BlobGame.ResourceHandling;

//public interface ITextureResource {
//    internal string Key { get; }
//    internal void DrawScreen(RectangleF destinationRect, Color? tint = null, Vector2? origin = null, float rotation = 0);
//    internal void DrawWorld(RectangleF destinationRect, Color? tint = null, Vector2? origin = null, float rotation = 0);
//}

public class TextureResource {
    public string Key { get; }

    private Texture Fallback { get; }
    private Func<string, Texture?> ResourceRetriever { get; }

    private Texture? _Reource { get; set; }
    public Texture Resource {
        get {
            if (_Reource == null)
                _Reource = ResourceRetriever(Key);

            return _Reource ?? Fallback;
        }
    }

    internal TextureResource(string key, Texture fallback, Func<string, Texture?> resourceRetriever) {
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

    internal void DrawWorld(RectangleF destinationRect, Color? tint, Vector2? origin, float rotation) {
        if (tint == null)
            tint = Raylib.WHITE;

        if (origin == null)
            origin = Vector2.Zero;

        Rectangle sourceRect = new Rectangle(0, 0, Resource.width, Resource.height);
        Raylib.DrawTexturePro(Resource, sourceRect, new Rectangle(destinationRect.Left, destinationRect.Bottom, destinationRect.Width, -destinationRect.Height), origin.Value, RayMath.RAD2DEG * rotation, tint.Value);
    }
}
