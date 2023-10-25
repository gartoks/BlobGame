using nkast.Aether.Physics2D.Dynamics;
using Raylib_CsLo;
using Vector2 = nkast.Aether.Physics2D.Common.Vector2;

namespace BlobGame.Game;
public abstract class GameObject : IEquatable<GameObject?> {
    public Guid Id { get; }
    public string Name { get; set; }

    public Vector2 Position => Body.Position * 10f;
    public float Rotation => Body.Rotation * 10f;
    public int ZIndex { get; set; }

    internal Body Body { get; }
    protected abstract Fixture Fixture { get; }

    protected internal GameObject(string name, World world, Vector2 position, float rotation, BodyType bodyType) {
        Id = Guid.NewGuid();
        Name = name;
        ZIndex = 0;

        Body = world.CreateBody(new Vector2(position.X / 10f, position.Y / 10f), rotation, bodyType);
        Body.Tag = this;
    }

    internal void Draw() {
        RlGl.rlPushMatrix();
        RlGl.rlTranslatef(Position.X, Position.Y, 0);
        RlGl.rlRotatef(RayMath.RAD2DEG * Rotation, 0, 0, 1);

        DrawInternal();

        RlGl.rlPopMatrix();
    }

    protected internal abstract void DrawInternal();

    public override bool Equals(object? obj) => Equals(obj as GameObject);
    public bool Equals(GameObject? other) => other is not null && Id.Equals(other.Id);
    public override int GetHashCode() => HashCode.Combine(Id);

    public static bool operator ==(GameObject? left, GameObject? right) => EqualityComparer<GameObject>.Default.Equals(left, right);
    public static bool operator !=(GameObject? left, GameObject? right) => !(left == right);
}
