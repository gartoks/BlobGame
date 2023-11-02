using BlobGame.App;
using BlobGame.Util;
using Raylib_CsLo;
using System.Numerics;

namespace BlobGame.Game.Gui;
internal abstract class InteractiveGuiElement : GuiElement {

    internal bool IsHovered => Bounds.Contains(Input.ScreenToWorld(Raylib.GetMousePosition()));

    protected InteractiveGuiElement(float x, float y, float w, float h, Vector2? pivot)
        : base(x, y, w, h, pivot) {
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
