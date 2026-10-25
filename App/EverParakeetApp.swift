import SwiftUI

@main
struct EverParakeetApp: App {
    @State private var model = AppModel()
    private let autotest = CommandLine.arguments.contains("-autotest")

    var body: some Scene {
        WindowGroup {
            if autotest {
                Text("Running self-test...")
                    .task {
                        await AutoTest.run()
                        exit(0)
                    }
            } else {
                ContentView()
                    .environment(model)
                    .task { await start() }
                    #if DEBUG
                    .task { if let id = AutoRun.requested { await AutoRun.go(model, id: id) } }
                    #endif
            }
        }
    }

    private func start() async {
        #if DEBUG
        if let scene = PreviewScene.requested {       // simulator screenshots: show a stubbed state, load nothing
            model.apply(scene)
            return
        }
        #endif
        await model.start()
    }
}
