using nkast.Aether.Physics2D.Dynamics;
using Raylib_CsLo;
using Vector2 = nkast.Aether.Physics2D.Common.Vector2;

namespace BlobGame.Game;
/// <summary>
/// Base class for all game objects included in the game simulation.
/// </summary>
public abstract class GameObject : IEquatable<GameObject?> {
    public const float POSITION_MULTIPLIER = 10f;

    /// <summary>
    /// The id uniqyely identifying this game object.
    /// </summary>
    public Guid Id { get; }
    /// <summary>
    /// The display name of the game object.
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// The position of the game object in arena coordinates.
    /// Since the physics engine works better with smaller values, the position is scaled by 10.
    /// </summary>
    public Vector2 Position => Body.Position * POSITION_MULTIPLIER;
    /// <summary>
    /// The rotation of the game object in radians.
    /// </summary>
    public float Rotation => Body.Rotation;
    /// <summary>
    /// The z position of the object. Objects with a higher z position are drawn on top of objects with a lower z position.
    /// </summary>
    public int ZIndex { get; set; }

    /// <summary>
    /// The physics engine body of the game object.
    /// </summary>
    internal Body Body { get; }
    /// <summary>
    /// The physics engine fixture of the game object.
    /// </summary>
    protected abstract Fixture Fixture { get; }

    protected internal GameObject(string name, World world, Vector2 position, float rotation, BodyType bodyType) {
        Id = Guid.NewGuid();
        Name = name;
        ZIndex = 0;

        Body = world.CreateBody(new Vector2(position.X / POSITION_MULTIPLIER, position.Y / POSITION_MULTIPLIER), rotation, bodyType);
        Body.Tag = this;
    }

    /// <summary>
    /// Draws the game object. Calls optional custom draw logic. Translates all drawing to the game object's position and rotation.
    /// </summary>
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
