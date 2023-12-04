using BlobGame.App;
using OpenTK.Mathematics;

namespace BlobGame.Util;
internal static class GuiBoundsParser {

    public static Box2 Parse(string bounds) {
        string[] split = bounds.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        if (split.Length != 4)
            throw new Exception("Invalid bounds string: " + bounds);

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

        if (!float.TryParse(split[0], out float x))
            throw new Exception("Invalid bounds string: " + bounds);

        if (!float.TryParse(split[1], out float y))
            throw new Exception("Invalid bounds string: " + bounds);

        if (!float.TryParse(split[2], out float w))
            throw new Exception("Invalid bounds string: " + bounds);

        if (!float.TryParse(split[3], out float h))
            throw new Exception("Invalid bounds string: " + bounds);

        x = xUsesPixel ? x : x * GameApplication.PROJECTION_WIDTH;
        y = yUsesPixel ? y : y * GameApplication.PROJECTION_HEIGHT;
        w = wUsesPixel ? w : w * GameApplication.PROJECTION_WIDTH;
        h = hUsesPixel ? h : h * GameApplication.PROJECTION_HEIGHT;

        return new Box2(x, y, x + w, y + h);
    }

}
