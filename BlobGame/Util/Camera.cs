using BlobGame.App;
using OpenTK.Mathematics;

namespace BlobGame.Util;
/// <summary>
/// Class For controlling the camera.
/// Is very bare bones, but translation, rotation etc, are not needed in the game
/// </summary>
internal sealed class Camera {
    /// <summary>
    /// The projection matrix for the camera.
    /// </summary>
    public Matrix4 ProjectionMatrix { get; }

    public Camera() {
        ProjectionMatrix = Matrix4.CreateOrthographicOffCenter(0, GameApplication.PROJECTION_WIDTH, GameApplication.PROJECTION_HEIGHT, 0, -1, 1);
    }

    public Vector2 NormalizedToViewport(Vector2 normalized) {
        return new Vector2(normalized.X * GameApplication.PROJECTION_WIDTH, normalized.Y * GameApplication.PROJECTION_HEIGHT);
    }

    /// <summary>
    /// Converts a view position to a world position.
    /// </summary>
    /// <param name="viewPos"></param>
    /// <returns></returns>
    public Vector2 ViewToWorld(Vector2 viewPos) {
        Matrix4 inv = Matrix4.Invert(ProjectionMatrix);
        Vector4 v = inv * new Vector4(viewPos.X, viewPos.Y, 0, 1);
        return new Vector2(v.X, v.Y);
    }

    /// <summary>
    /// Converts a world position to a view position.
    /// </summary>
    /// <param name="worldPos"></param>
    /// <returns></returns>
    public Vector2 WorldToView(Vector2 worldPos) {
        Vector4 v = ProjectionMatrix * new Vector4(worldPos.X, worldPos.Y, 0, 1);
        return new Vector2(v.X, v.Y);
    }
}
