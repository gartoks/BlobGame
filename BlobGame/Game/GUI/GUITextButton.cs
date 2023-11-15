using BlobGame.App;
using BlobGame.ResourceHandling;
using BlobGame.ResourceHandling.Resources;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal sealed class GuiTextButton : InteractiveGuiElement {
    public NPatchTextureResource BaseTexture { get; set; }
    public NPatchTextureResource SelectedTexture { get; set; }

    public GuiLabel Label { get; }

    public bool IsClicked { get; private set; }

    public GuiTextButton(string boundsString, string text, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), text, pivot) {
    }

    private GuiTextButton(Rectangle bounds, string text, Vector2? pivot = null)
        : this(bounds.X, bounds.Y, bounds.width, bounds.height, text, pivot) {
    }

    public GuiTextButton(float x, float y, float w, float h, string text, Vector2? pivot = null)
        : base(x, y, w, h, pivot) {

        BaseTexture = ResourceManager.NPatchLoader.Get("button_up");
        SelectedTexture = ResourceManager.NPatchLoader.Get("button_selected");
        //Panel = new GuiPanel(Bounds, new Vector2(0, 0));
        Label = new GuiLabel(Bounds, text, new Vector2(0, 0));
    }

    protected override void DrawInternal() {
        NPatchTextureResource texture = BaseTexture;
        if (IsHovered)
            texture = SelectedTexture;

        IsClicked = (IsHovered && Input.IsMouseButtonActive(MouseButton.MOUSE_BUTTON_LEFT)) ||
            (GuiManager.HasFocus(this) && Input.IsHotkeyActive("confirm"));

        if (IsClicked)
            Focus();

        if (HasFocus())
            texture = SelectedTexture;

        texture.Draw(Bounds, Vector2.Zero, Raylib.WHITE);

        Label.Draw();

        Input.WasMouseHandled[MouseButton.MOUSE_BUTTON_LEFT] |= IsClicked;
    }

}
