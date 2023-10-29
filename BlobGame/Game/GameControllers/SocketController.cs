using BlobGame.Game.GameObjects;
using BlobGame.Game.Scenes;
using System.Diagnostics;
using System.Net.Sockets;

namespace BlobGame.Game.GameControllers;

internal class SocketController : IGameController {
    /// <summary>
    /// Represents the game scene associated with this controller.
    /// </summary>
    private GameScene Scene { get; }
    
    /// <summary>
    /// The tcp client used to control this controller.
    /// </summary>
    private TcpClient client;
    /// <summary>
    /// The network stream used to communicate with the server.
    /// </summary>
    private NetworkStream stream;

    private (float t, bool shouldDrop) frameInputs;

    public SocketController(GameScene scene) {
        Scene = scene;

        try{
            client = new TcpClient("localhost", 1234);
            stream = client.GetStream();
        }
        catch(SocketException){
            Debug.WriteLine("Controller stream was closed.");
            GameManager.SetScene(new MainMenuScene());
        }
        Debug.WriteLine("Connected to localhost:1234");
    }

    ~SocketController(){
        Close();
    }

    public void Close(){
        Debug.WriteLine("Closing tcp socket");
        client.Close();
    }

    /// <summary>
    /// Retrieves the current value of t, which represents the position of the dropper above the arena.
    /// </summary>
    /// <returns>The current value of t.</returns>
    public float GetCurrentT() {
        return frameInputs.t;
    }

    /// <summary>
    /// Attempts to spawn a blob in the provided game simulation.
    /// </summary>
    /// <param name="simulation">The game simulation in which to spawn the blob.</param>
    /// <param name="t">The t value at which the blob is spawned, which represents the position of the dropper above the arena..</param>
    /// <returns>True if blob spawning was attempted, otherwise false.</returns>
    public bool SpawnBlob(ISimulation simulation, out float t) {
        t = -1;
        if (!simulation.CanSpawnBlob)
            return false;
        
        t = GetCurrentT();
        
        return frameInputs.shouldDrop;
    }
    public void Update(ISimulation simulation){
        // send simulation state
        SendGameState(simulation);

        // wait for inputs
        ReceiveInputs();
    }

    public void SendGameState(ISimulation simulation){
        

        List<Blob> blobs = new();
        simulation.GameObjects.Enumerate(go => {
            if (go.GetType() == typeof(Blob)){
                blobs.Add((Blob)go);
            }
        });

        IEnumerable<byte> buffer = new byte[0];
        buffer = buffer.Concat(BitConverter.GetBytes(blobs.Count));
        foreach(Blob blob in blobs){
            buffer = buffer.Concat(BitConverter.GetBytes(blob.Position.X));
            buffer = buffer.Concat(BitConverter.GetBytes(blob.Position.Y));
            buffer = buffer.Concat(BitConverter.GetBytes((int)blob.Type));
        }
        buffer = buffer.Concat(BitConverter.GetBytes((int)simulation.CurrentBlob));
        buffer = buffer.Concat(BitConverter.GetBytes((int)simulation.NextBlob));
        buffer = buffer.Concat(BitConverter.GetBytes(simulation.CanSpawnBlob));

        bool failed = false;
        try{
            stream.Write(buffer.ToArray());
        }
        catch(SocketException){
            failed = true;
        }
        catch(IOException){
            failed = true;
        }

        if (failed){
            Debug.WriteLine("Controller stream was closed.");
            GameManager.SetScene(new MainMenuScene());
            return;
        }
    }

    /// <summary>
    /// Waits for the next frame of inputs and stores them in frameInputs.
    /// </summary>
    public void ReceiveInputs(){
        byte[] buffer = new byte[5];
        bool failed = false;
        try{
            stream.ReadExactly(buffer, 0, 5);
        }
        catch(EndOfStreamException){
            failed = true;
        }
        catch(SocketException){
            failed = true;
        }
        catch(IOException){
            failed = true;
        }
        if (failed){
            Debug.WriteLine("Controller input stream was closed.");
            GameManager.SetScene(new MainMenuScene());
            return;
        }

        float t = BitConverter.ToSingle(buffer, 0);
        bool shouldDrop = BitConverter.ToBoolean(buffer, 4);
        frameInputs = (t, shouldDrop);
    }


}
