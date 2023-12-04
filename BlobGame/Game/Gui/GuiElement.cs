using OpenTK.Mathematics;

namespace BlobGame.Game.Gui;
internal abstract class GuiElement : IEquatable<GuiElement?> {
    private Guid Id { get; }

    protected Vector2 Pivot { get; }
    protected Box2 Bounds { get; }
    protected int ZIndex { get; }

    public bool Enabled { get; set; }

    protected GuiElement(Vector2 position, Vector2 size, Vector2? pivot, int zIndex)
        : this(position.X, position.Y, size.X, size.Y, pivot, zIndex) { }

    protected GuiElement(float x, float y, float w, float h, Vector2? pivot, int zIndex) {
        Id = Guid.NewGuid();
        Pivot = pivot ?? Vector2.Zero;

        x += -w * Pivot.X;
        y += -h * Pivot.Y;
        Bounds = new Box2(x, y, x + w, y + h);
        ZIndex = zIndex;

        Enabled = true;

    }

    internal virtual void Load() {
        GuiManager.AddElement(this);
    }

    internal void Unload() {
    }

    internal void Draw() {
        if (!Enabled)
            return;

        DrawInternal();
    }

    protected abstract void DrawInternal();

    public override bool Equals(object? obj) => Equals(obj as GuiElement);
    public bool Equals(GuiElement? other) => other is not null && Id.Equals(other.Id);
    public override int GetHashCode() => HashCode.Combine(Id);

    public static bool operator ==(GuiElement? left, GuiElement? right) => EqualityComparer<GuiElement>.Default.Equals(left, right);
    public static bool operator !=(GuiElement? left, GuiElement? right) => !(left == right);
}
