using BlobGame.App;
using BlobGame.Game.Scenes;
using Raylib_CsLo;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net.Sockets;
using System.Numerics;
using System.Security.Cryptography.X509Certificates;

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

    /// <summary>
    /// Tells the receiver thread to stop.
    /// </summary>
    private bool IsClosed = false;

    /// <summary>
    /// Receives frame data on a different thread.
    /// </summary>
    private Thread receiverThread;

    private ConcurrentQueue<(float t, bool shouldDrop)> frameInputs;

    private object inputLock = new object();

    public SocketController(GameScene scene) {
        Scene = scene;

        client = new TcpClient("localhost", 1234);
        stream = client.GetStream();
        Debug.WriteLine("Connected to localhost:1234");
    
        frameInputs = new();
        frameInputs.Enqueue((0.5f, false));

        receiverThread = new Thread(ReceiveFrames);
        receiverThread.Start();
    }

    ~SocketController(){
        Close();
    }

    public void Close(){
        IsClosed = true;
        Debug.WriteLine("Waiting for receiver thread");
        receiverThread.Join();
        Debug.WriteLine("Closing tcp socket");
        client.Close();
    }

    /// <summary>
    /// Retrieves the current value of t, which represents the position of the dropper above the arena.
    /// </summary>
    /// <returns>The current value of t.</returns>
    public float GetCurrentT() {
        return frameInputs.Last().t;
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
        
        return frameInputs.Last().shouldDrop;
    }

    /// <summary>
    /// Waits for the next frame of inputs and stores them in nextFrameInputs.
    /// </summary>
    public void ReceiveFrames(){
        try{
            while (!IsClosed){
                while (frameInputs.Count > 2){
                    frameInputs.TryDequeue(out _);
                }

                byte[] buffer = new byte[1024];
                
                stream.ReadExactly(buffer, 0, 5);
                float t = BitConverter.ToSingle(buffer, 0);
                bool shouldDrop = BitConverter.ToBoolean(buffer, 4);
                frameInputs.Enqueue((t, shouldDrop));
            }
        }
        catch(EndOfStreamException){
            Debug.WriteLine("Controller stream was closed.");
            GameManager.SetScene(new MainMenuScene());
        }
        catch(SocketException){
            Debug.WriteLine("Controller stream was closed.");
            GameManager.SetScene(new MainMenuScene());
        }
    }

}
