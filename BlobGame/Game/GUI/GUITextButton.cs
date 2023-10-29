using BlobGame.App;
using BlobGame.ResourceHandling;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GuiTextButton {
    private GuiPanel Panel { get; }
    private GuiLabel Label { get; }
    private Rectangle Bounds { get; }

    public GuiTextButton(Vector2 pos, Vector2 size, string text, Vector2? pivot = null)
        : this(pos.X, pos.Y, size.X, size.Y, text, pivot) {
    }

    public GuiTextButton(float x, float y, float w, float h, string text, Vector2? pivot = null) {
        if (pivot != null) {
            x += -w * pivot.Value.X;
            y += -h * pivot.Value.Y;
        }

        Panel = new GuiPanel(x, y, w, h, ResourceManager.GetColor("light_accent"), new Vector2(0, 0));
        Label = new GuiLabel(x, y, w, h, text, new Vector2(0, 0));

        Bounds = new Rectangle(x, y, w, h);
    }

    internal bool Draw() {
        bool containsMouse = Bounds.Contains(Input.ScreenToWorld(Raylib.GetMousePosition()));
        ColorResource bgColor = ResourceManager.GetColor("light_accent");
        if (containsMouse)
            bgColor = ResourceManager.GetColor("dark_accent");

        Panel.Color = bgColor;
        Panel.Draw();
        Label.Draw();

        bool clicked = containsMouse && Input.IsMouseButtonActive(MouseButton.MOUSE_BUTTON_LEFT);
        Input.WasMouseHandled[MouseButton.MOUSE_BUTTON_LEFT] |= clicked;

        return clicked;
    }

}
