using Raylib_CsLo;
using System.Globalization;

namespace BlobGame.Util;
internal static class GuiBoundsParser {

    public static Rectangle Parse(string bounds) {
        bounds = bounds.Trim().Replace(",", ".");

        string[] split = bounds.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        bool xUsesPixel = false;
        if (split[0].EndsWith("px")) {
            xUsesPixel = true;
            split[0] = split[0][..^2];
        }

        bool yUsesPixel = false;
        if (split[1].EndsWith("px")) {
            yUsesPixel = true;
            split[1] = split[1][..^2];
        }

        bool wUsesPixel = false;
        if (split[2].EndsWith("px")) {
            wUsesPixel = true;
            split[2] = split[2][..^2];
        }

        bool hUsesPixel = false;
        if (split[3].EndsWith("px")) {
            hUsesPixel = true;
            split[3] = split[3][..^2];
        }

        if (split.Length != 4)
            throw new Exception($"Invalid bounds string (length): '{bounds}'");

        if (!float.TryParse(split[0], CultureInfo.InvariantCulture, out float x))
            throw new Exception($"Invalid bounds string (x): '{bounds}'");

        if (!float.TryParse(split[1], CultureInfo.InvariantCulture, out float y))
            throw new Exception($"Invalid bounds string (y): '{bounds}'");

        if (!float.TryParse(split[2], CultureInfo.InvariantCulture, out float w))
            throw new Exception($"Invalid bounds string (w): '{bounds}'");

        if (!float.TryParse(split[3], CultureInfo.InvariantCulture, out float h))
            throw new Exception($"Invalid bounds string (h): '{bounds}'");

        x = xUsesPixel ? x : x * Application.BASE_WIDTH;
        y = yUsesPixel ? y : y * Application.BASE_HEIGHT;
        w = wUsesPixel ? w : w * Application.BASE_WIDTH;
        h = hUsesPixel ? h : h * Application.BASE_HEIGHT;

        return new Rectangle(x, y, w, h);
    }

}
