using BlobGame.Util;
using OpenTK.Mathematics;

namespace BlobGame.Game.Gui;
internal class GuiMessageBox : GuiElement {
    private string AcceptButtonText { get; }
    private string DeclineButtonText { get; }
    private Action AcceptAction { get; }
    private Action DeclineAction { get; }

    private GuiPanel BackgroundPanel { get; }
    private GuiLabel MessageLabel { get; }
    private GuiTextButton AcceptButton { get; }
    private GuiTextButton DeclineButton { get; }

    public GuiMessageBox(string boundsString, string message, string acceptButtonText, string declineButtonText, Action acceptAction, Action declineAction, int zIndex, Vector2? pivot = null)
        : this(GuiBoundsParser.Parse(boundsString), message, acceptButtonText, declineButtonText, acceptAction, declineAction, zIndex, pivot) {
    }

    private GuiMessageBox(Box2 bounds, string message, string acceptButtonText, string declineButtonText, Action acceptAction, Action declineAction, int zIndex, Vector2? pivot = null)
        : this(bounds.X(), bounds.Y(), bounds.Width(), bounds.Height(), message, acceptButtonText, declineButtonText, acceptAction, declineAction, zIndex, pivot) {
    }

    public GuiMessageBox(float x, float y, float w, float h, string message, string acceptButtonText, string declineButtonText, Action acceptAction, Action declineAction, int zIndex, Vector2? pivot)
        : base(x, y, w, h, pivot, zIndex) {

        AcceptButtonText = acceptButtonText;
        DeclineButtonText = declineButtonText;
        AcceptAction = acceptAction;
        DeclineAction = declineAction;

        BackgroundPanel = new GuiPanel(Bounds.X(), Bounds.Y(), Bounds.Width(), Bounds.Height(), ZIndex);
        MessageLabel = new GuiLabel(Bounds.X() + Bounds.Width() / 2, Bounds.Y(), Bounds.Width(), Bounds.Height() / 2, message, ZIndex + 1, new Vector2(0.5f, 0));

    }

    internal override void Load() {
        BackgroundPanel.Load();
        MessageLabel.Load();
        AcceptButton.Load();
        DeclineButton.Load();
    }

    protected override void DrawInternal() {

    }
}
