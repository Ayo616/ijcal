def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--ood_type', type=str, default='structure', choices=['structure', 'label', 'feature'],
                        help='only for cora/amazon/arxiv datasets')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_prop', type=float, default=.1,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.1,
                        help='validation label proportion')
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--epochs', type=int, default=400)

    # model network
    parser.add_argument('--method', type=str, default='GEVN', choices=['msp', 'GEVN','OE','ODIN','Mahalanobis'])
    parser.add_argument('--backbone', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for GNN classifiers')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    parser.add_argument('--hops', type=int, default=2,
                        help='power of adjacency matrix for sgc')

    # GEVN hyper
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')
    parser.add_argument('--use_reg',action='store_true', help='whether to use energy regularization loss')
    parser.add_argument('--lamda', type=float, default=1.0, help='weight for regularization')
    parser.add_argument('--K', type=int, default=2, help='number of layers for energy belief propagation')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual connection in propagation')
    parser.add_argument('--use_mlayer',action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--use_sprop',action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--use_srece',action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--oodloss', action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--reduce', type=float,default=1, help='whether to use energy belief propagation')

    parser.add_argument('--record', default='Model', choices=['Model', 'Model_mlayer', 'Model_slayer','Model_oodloss','Model_Final'])


    # baseline hyper
    parser.add_argument('--noise', type=float, default=0., help='param for baseline ODIN and Mahalanobis')

    # training
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dropout1', type=float, default=0.2)
    parser.add_argument('--dropout3', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--use_bn', action='store_true', help='use batch norm')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--print_args', action='store_true',
                        help='print args for hyper-parameter searching')
    parser.add_argument('--mode', type=str, default='detect', choices=['classify', 'detect'])