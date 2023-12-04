using BlobGame.App;
using BlobGame.Util;
using nkast.Aether.Physics2D.Common;
using nkast.Aether.Physics2D.Dynamics;
using SimpleGL.Graphics.Rendering;

namespace BlobGame.Game;
/// <summary>
/// Class for the arena's walls.
/// </summary>
internal sealed class Wall : GameObject {
    /// <summary>
    /// The physics engine fixture attached to the wall's body.
    /// </summary>
    protected override Fixture Fixture { get; }

    /// <summary>
    /// The width of the wall.
    /// </summary>
    private float Width { get; }
    /// <summary>
    /// The height of the wall.
    /// </summary>
    private float Height { get; }

    /// <summary>
    /// Creates a new wall with the given parameters.
    /// </summary>
    /// <param name="name">The name of wall</param>
    /// <param name="world">The world to add the physics engine body to.</param>
    /// <param name="bounds">The bounds of the wall. Used to position and scale it.</param>
    public Wall(string name, World world, OpenTK.Mathematics.Box2 bounds)
        : base(name, world, new Vector2(bounds.X(), bounds.Y()), 0, BodyType.Static) {
        Width = bounds.Width();
        Height = bounds.Height();

        Fixture = Body.CreateRectangle(Width / 10f, Height / 10f, 1, new Vector2(Width / 10f / 2, Height / 10f / 2));
        Fixture.Restitution = 0;
        Fixture.Friction = 0.15f;
        ZIndex = 0;
    }

    /// <summary>
    /// Custom wall drawing logic.
    /// </summary>
    public override void Render(OpenTK.Mathematics.Vector2 offset) {
        if (GameApplication.DRAW_DEBUG)
            Primitives.DrawRectangleLines(
                OpenTK.Mathematics.Vector2.Zero,
                new OpenTK.Mathematics.Vector2(Width, Height),
                2f,
                OpenTK.Mathematics.Vector2.Zero,
                0,
                100,
                OpenTK.Mathematics.Color4.White);
    }
}
