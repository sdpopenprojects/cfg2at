# cfg2at

CFG2AT: Control Flow Graph and Graph Attention Network based Software Defect Prediction

We propose a novel SDP approach called Control Flow Graph and Graph Attention Network based Software Defect Prediction (CFG2AT). CFG2AT is designed to identify software defects automatically and contains a graph-structured attention unit to capture control flow information effectively.

## Project Structure

- `data` : The datasets of CFG2AT for Java and Python languages. It contains `graph `dir with control flow graph information dir and tradition dir with original *Promise* and *JIRA* datasets.
- `.py` : The scripts of CFG2AT.

## Environment

Install required packages:

```shell
conda env create -f cfg2at.yaml
conda activate cfg2at
pip install -r requirements.txt
```

## Train and Test

#### Usage

```txt
usage: mainforjava.py [-h] [--runTimes RUNTIMES] [--layers LAYERS] [--hiddens HIDDENS] [--epochs EPOCHS] [--numHeads NUMHEADS]
                      {ant,activemq,lucene,jruby,hbase,hive} trainVersion {ant,activemq,lucene,jruby,hbase,hive} testVersion

usage: mainforpy.py [-h] [--runTimes RUNTIMES] [--layers LAYERS] [--hiddens HIDDENS] [--epochs EPOCHS] [--numHeads NUMHEADS]
                    {pandas} trainVersion {pandas} testVersion
```

#### For Example

If perform WPDP for Java projects, you run:

  `python mainforjava.py ant 1.5 ant 1.6`

If perform CPDP for Java projects, you run:

 `python mainforjava.py hbase 0.95.0 activemq 5.3.0`

If perform WPDP for Python projects, you run

 `python mainforpy.py pandas 2.2.0 pandas 2.2.1`
