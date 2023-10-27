using BlobGame.App;
using BlobGame.Drawing;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GUITextButton {
    private GUIPanel Panel { get; }
    private GuiLabel Label { get; }
    private Rectangle Bounds { get; }

    public GUITextButton(Vector2 pos, Vector2 size, string text, Vector2? pivot = null)
        : this(pos.X, pos.Y, size.X, size.Y, text, pivot) {
    }

    public GUITextButton(float x, float y, float w, float h, string text, Vector2? pivot = null) {
        if (pivot != null) {
            x += -w * pivot.Value.X;
            y += -h * pivot.Value.Y;
        }

        Panel = new GUIPanel(x, y, w, h, Renderer.MELBA_LIGHT_PINK, new Vector2(0, 0));
        Label = new GuiLabel(x, y, w, h, text, new Vector2(0, 0));

        Bounds = new Rectangle(x, y, w, h);
    }

    internal bool Draw() {
        bool containsMouse = Bounds.Contains(Input.ScreenToWorld(Raylib.GetMousePosition()));
        Color bgColor = Renderer.MELBA_LIGHT_PINK;
        if (containsMouse)
            bgColor = Renderer.MELBA_DARK_PINK;

        Panel.Color = bgColor;
        Panel.Draw();
        Label.Draw();

        return containsMouse && Input.IsMouseButtonActive(MouseButton.MOUSE_BUTTON_LEFT);
    }

}
