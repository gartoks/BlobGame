using BlobGame.App;
using OpenTK.Mathematics;

namespace BlobGame.Game.Gui;
internal abstract class InteractiveGuiElement : GuiElement {

    internal bool IsHovered => Bounds.ContainsInclusive(Input.GetMousePosition());

    protected InteractiveGuiElement(float x, float y, float w, float h, int zIndex, Vector2? pivot)
        : base(x, y, w, h, pivot, zIndex) {
    }

    internal void Focus() {
        GuiManager.Focus(this);
    }

    internal void Unfocus() {
        if (HasFocus())
            GuiManager.Focus(null);
    }

    internal bool HasFocus() {
        return GuiManager.HasFocus(this);
    }
}
