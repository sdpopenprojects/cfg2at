from mygatv2 import MyGATv2
from myprojectutils import MyUtils

import argparse
import os
import pandas as pd

os.environ["DGLBACKEND"] = "pytorch"

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics

import dgl
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset


class CFGDataset(DGLDataset):
    def __init__(
        self,
        trainProjectName,
        trainVersion,
        testProjectName,
        testVersion,
        isTraining,
    ):
        self.instruction = {
            # core statements
            "soot.jimple.internal.JNopStmt": 1,
            "soot.jimple.internal.JIdentityStmt": 2,
            "soot.jimple.internal.JAssignStmt": 3,
            ## intraprocedural control
            "soot.jimple.internal.JIfStmt": 4,
            "soot.jimple.internal.JGotoStmt": 5,
            "soot.jimple.internal.JTableSwitchStmt": 6,
            "soot.jimple.internal.JLookupSwitchStmt": 7,
            ## interprocedural control
            "soot.jimple.internal.JInvokeStmt": 8,
            "soot.jimple.internal.JReturnStmt": 9,
            "soot.jimple.internal.JReturnVoidStmt": 10,
            ## monitor statements
            "soot.jimple.internal.JEnterMonitorStmt": 11,
            "soot.jimple.internal.JExitMonitorStmt": 12,
            ## others
            "soot.jimple.internal.JThrowStmt": 13,
            "soot.jimple.internal.JRetStmt": 14,
            # "soot.jimple.internal.JBreakPointStmt": 15, for debug, not use for real projects
        }
        self.instruction_enc = F.one_hot(torch.arange(len(self.instruction) + 1))

        self.project, self.version = (
            (trainProjectName, trainVersion)
            if isTraining
            else (testProjectName, testVersion)
        )
        super().__init__(name="CFG")

    def process(self):
        nodes = pd.read_csv(
            "data/java/graph/{}/{}/nodes.csv".format(self.project, self.version)
        )
        edges = pd.read_csv(
            "data/java/graph/{}/{}/edges.csv".format(self.project, self.version)
        )
        properties = pd.read_csv(
            "data/java/graph/{}/{}/properties.csv".format(self.project, self.version)
        )
        self.graphs = []
        self.labels = []

        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row["cfg_id"]] = 1 if row["label"] >= 1 else 0
            num_nodes_dict[row["cfg_id"]] = row["num_nodes"]
        edges_group = edges.groupby("cfg_id")
        nodes_group = nodes.groupby("cfg_id")
        for graph_id in edges_group.groups:
            edges_of_id = edges_group.get_group(graph_id)
            nodes_of_id = nodes_group.get_group(graph_id).sort_values(by="node_id")
            src = edges_of_id["src"].to_numpy()
            dst = edges_of_id["dst"].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g.ndata["instruction_type"] = torch.stack(
                list(
                    map(
                        lambda x: (
                            self.instruction_enc[self.instruction[x]]
                            if self.instruction[x]
                            else self.instruction_enc[0]
                        ),
                        nodes_of_id["instruction_type"].values.tolist(),
                    )
                )
            ).float()
            self.graphs.append(g)
            self.labels.append(label)

        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


def evaluate(net, data_iter, device):
    if isinstance(net, torch.nn.Module):
        net.eval()
    net.to(device)

    y, y_hat = [], []
    labels, preds = [], []

    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)

            preds.append(y_hat.argmax(1))
            labels.append(y)

        return (
            torchmetrics.functional.precision(
                torch.cat(preds), torch.cat(labels), "binary"
            ),
            torchmetrics.functional.recall(
                torch.cat(preds), torch.cat(labels), "binary"
            ),
            torchmetrics.functional.fbeta_score(
                torch.cat(preds), torch.cat(labels), "binary", beta=1.0
            ),
            torchmetrics.functional.auroc(
                torch.cat(preds).float(), torch.cat(labels), "binary"
            ),
            torchmetrics.functional.matthews_corrcoef(
                torch.cat(preds).float(), torch.cat(labels), "binary"
            ),
        )


def train(net, train_iter, num_epochs, updater, scheduler, device):
    net.apply(MyUtils.weight_init)
    weight = MyUtils.class_weight(train_iter.dataset, device)
    net.to(device)
    for epoch in range(num_epochs):
        train_loss, train_acc = MyUtils.trainEpoch(
            net,
            train_iter,
            F.cross_entropy,
            weight,
            updater,
            device,
        )
        scheduler.step()
        if epoch % 10 == 0:
            print(
                "Epoch {}: train loss {:.4f}  train acc: {:.5f} \n".format(
                    epoch, train_loss, train_acc
                ),
                flush=True,
            )


def sdpProcess(
    trainProjectName,
    testProjectName,
    trainVersion,
    testVersion,
    device,
    layers,
    hiddens,
    par_epochs,
    par_num_heads,
):
    lr, epochs, batch_size, dropout = (1e-3, par_epochs, 64, 0.2)
    schedulerStepSize, schedulerGamma = 16, 0.9
    hidden_dim, num_heads = hiddens, par_num_heads

    model = MyGATv2(hidden_dim, 2, num_heads, dropout, dropout, layers)

    trainDataset, testDataset = CFGDataset(
        trainProjectName, trainVersion, testProjectName, testVersion, True
    ), CFGDataset(testProjectName, testVersion, testProjectName, testVersion, False)

    trainDataloader, testDataloader = GraphDataLoader(
        trainDataset, batch_size=batch_size, drop_last=False
    ), GraphDataLoader(testDataset, batch_size=batch_size, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=schedulerStepSize, gamma=schedulerGamma
    )

    print(
        "training start: {}--{}-----{}--{}\n".format(
            trainProjectName, trainVersion, testProjectName, testVersion
        ),
        flush=True,
    )

    train(
        model,
        trainDataloader,
        epochs,
        optimizer,
        scheduler,
        device,
    )

    metric_all = evaluate(model, testDataloader, device)

    print(
        "===========P=============\n{}\n===========R=============\n{}\n===========F1=============\n{}\n===========AUC=============\n{}\n===========MCC=============\n{}\n".format(
            metric_all[0].item(),
            metric_all[1].item(),
            metric_all[2].item(),
            metric_all[3].item(),
            metric_all[4].item(),
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trainProjectName",
        choices=["ant", "activemq", "lucene", "jruby", "hbase", "hive"],
    )
    parser.add_argument("trainVersion", type=str)
    parser.add_argument(
        "testProjectName",
        choices=["ant", "activemq", "lucene", "jruby", "hbase", "hive"],
    )
    parser.add_argument("testVersion", type=str)
    parser.add_argument("--runTimes", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--hiddens", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--numHeads", type=int)
    args = parser.parse_args()

    (
        trainProjectName,
        testProjectName,
        trainVersion,
        testVersion,
        runTimes,
        layers,
        hiddens,
        epochs,
        num_heads,
    ) = (
        args.trainProjectName,
        args.testProjectName,
        args.trainVersion,
        args.testVersion,
        args.runTimes if args.runTimes != None else 1,
        args.layers if args.layers != None else 7,
        args.hiddens if args.hiddens != None else 32,
        args.epochs if args.epochs != None else 500,
        args.numHeads if args.numHeads != None else 5,
    )

    for i in range(runTimes):
        sdpProcess(
            trainProjectName,
            testProjectName,
            trainVersion,
            testVersion,
            MyUtils.try_gpu(),
            layers,
            hiddens,
            epochs,
            num_heads,
        )
