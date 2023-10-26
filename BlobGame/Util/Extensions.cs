using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Util;
internal static class Extensions {

    public static bool Contains(this Rectangle rect, Vector2 pos) {
        return pos.X >= rect.x && pos.X <= rect.x + rect.width && pos.Y >= rect.y && pos.Y <= rect.y + rect.height;
    }

    public static Color ChangeAlpha(this Color c, int alpha) {
        return new Color(c.r, c.g, c.b, alpha);
    }
}
