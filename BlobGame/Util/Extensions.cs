using OpenTK.Mathematics;

namespace BlobGame.Util;
internal static class Extensions {

    /*public static bool Contains(this Rectangle rect, Vector2 pos) {
        return pos.X >= rect.x && pos.X <= rect.x + rect.width && pos.Y >= rect.y && pos.Y <= rect.y + rect.height;
    }*/

    public static float X(this Box2 box) {
        return box.Min.X;
    }

    public static float Y(this Box2 box) {
        return box.Min.Y;
    }

    public static float Width(this Box2 box) {
        return box.Size.X;
    }

    public static float Height(this Box2 box) {
        return box.Size.Y;
    }

    public static Color4 ChangeAlpha(this Color4 c, int alpha) {
        return new Color4(c.R, c.G, c.B, alpha / 255f);
    }
}
