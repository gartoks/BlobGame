using BlobGame.App;
using BlobGame.Drawing;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.GUI;
internal static class GUITextButton {

    internal static bool Draw(Vector2 pos, Vector2 size, string text, Vector2? pivot = null) => Draw(pos.X, pos.Y, size.X, size.Y, text, pivot);
    internal static bool Draw(float x, float y, float w, float h, string text, Vector2? pivot = null) {

        x = x - w / 2;
        y = y - h / 2;
        if (pivot != null) {
            x += w * pivot.Value.X;
            y += h * pivot.Value.Y;
        }

        Rectangle rect = new Rectangle(x, y, w, h);
        bool containsMouse = rect.Contains(Input.ScreenToWorld(Raylib.GetMousePosition()));
        Color bgColor = Renderer.LIGHT_PINK;
        if (containsMouse)
            bgColor = Renderer.DARK_PINK;

        Raylib.DrawRectangleRounded(rect, 0.15f, 10, bgColor);
        Raylib.DrawRectangleRoundedLines(rect, 0.15f, 10, 8, Raylib.WHITE);
        int fontSize = 80;
        Raylib.DrawText(text, (int)(x + w / 2 - Raylib.MeasureText(text, fontSize) / 2), (int)(y + h / 2 - fontSize / 2), fontSize, Raylib.WHITE);

        return containsMouse && Input.IsMouseButtonActive(MouseButton.MOUSE_BUTTON_LEFT);
    }

}
