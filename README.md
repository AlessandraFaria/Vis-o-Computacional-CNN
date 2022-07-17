# Visão-Computacional-CNN
Execução de métodos de compute vision para classificação de imagens e encode via VAE

A visão humana é incrivelmente bonita e complexa, tudo começou há bilhões de anos, onde pequenos organismos desenvolveram uma mutação que os tornou sensíveis à luz, avançando rapidamente para hoje e há uma abundância de vida no planeta, todos com sistemas visuais muito semelhantes, incluindo os olhos para capturar receptores de luz no cérebro para acessá-lo e um córtex visual para processá-lo geneticamente modificado e equilibrado

Partes de um sistema que nos ajuda a fazer coisas tão simples como apreciar o nascer do sol, mas isso é apenas o começo nos últimos 30 anos, fizemos ainda mais progressos para estender essa incrível capacidade visual não apenas para nós mesmos, mas também para as máquinas. O primeiro tipo de câmera fotográfica foi inventado por volta de 1816, onde uma pequena caixa continha um pedaço de papel revestido com cloreto de prata quando o obturador era aberto, o cloreto de prata escurecia



Conseguimos imitar de perto como o olho humano pode capturar uma luz colorida, mas está acontecendo que isso foi a parte fácil entender o que está na foto é muito mais difícil. Com milhões de anos de contexto evolutivo podemos imediatamente entender a imagem de uma flor, mas um computador não tem a mesma vantagem de um algoritmo a imagem se parece com isso apenas uma enorme variedade de valores inteiros que representam intensidades em todo o espectro de cores não há contexto aqui, apenas uma pilha enorme de dados, verifica-se que o contexto é o cerne de obter algoritmos para entender o conteúdo da imagem da mesma maneira que o cérebro humano para fazer isso funcionar usamos um algoritmo muito parecido com o funcionamento do cérebro humano usando machine learning machine learning nos permite treinar efetivamente o contexto para um conjunto de dados para que um algoritmo possa entender o que todos aqueles números em uma organização específica realmente representam e se tivermos imagens que são difíceis para um humano classificar, o aprendizado de máquina pode alcançar uma precisão melhor.


Cães pastores e esfregões, onde é muito difícil até mesmo para nós diferenciarmos entre os dois com o modelo de aprendizado de máquina, podemos tirar várias imagens de cães pastores e esfregões e, desde que forneçamos dados suficientes, ele será capaz de dizer corretamente a diferença entre os dois.


visão computacional está assumindo desafios cada vez mais complexos e está vendo uma precisão que rivaliza com humanos realizando as mesmas tarefas de reconhecimento de imagem.

![image](https://user-images.githubusercontent.com/14276167/179123469-3c203033-199d-4472-8d39-61a02de81f6d.png)

Esses modelos não são perfeitos eles às vezes cometem erros o tipo específico de rede neural que realiza isso é chamado de rede neural convolucional ou CNN

CNN trabalha dividindo uma imagem em grupos menores de pixels chamados filtro cada filtro é uma matriz de pixels e a rede faz uma série de cálculos nesses pixels comparando-os com pixels em padrões específicos que a rede está procurando na primeira camada de uma CNN que é capaz de detectar padrões de alto nível como arestas e curvas à medida que a rede executa mais circunvoluções, pode começar a identificar objetos específicos, como rostos e animais, como uma CNN sabe o que procurar e se sua previsão é precisa, isso é feito por meio de um grande quantidade de dados de treinamento rotulados quando a CNN inicia todos os valores do filtro são randomizados, como resultado, suas previsões iniciais fazem pouco sentido cada vez que a CNN faz uma previsão em relação a dados rotulados, ele usa uma função de erro para comparar o quão perto sua previsão estava do rótulo real da imagem com base nesse erro ou em uma função de perda, a CNN atualiza seus valores de filtro e inicia o processo novamente, idealmente, cada iteração é executada com um pouco mais de precisão

![image](https://user-images.githubusercontent.com/14276167/179123521-48f356e8-57f3-41f2-95dc-e6c7839a6d2f.png)

o RNN processa cada sequência, usa uma função de perda ou erro para comparar sua saída prevista com a etiqueta correta, então ajusta os pesos e processa a sequência novamente até atingir uma precisão maior

# Aplicação de métodos de visão computacional
## Base de dados utilizada

O banco de dados MNIST está disponível em http://yann.lecun.com/exdb/mnist/

O banco de dados MNIST é um conjunto de dados de dígitos manuscritos. Possui 60.000 amostras de treinamento e 10.000 amostras de teste. Cada imagem é representada por 28x28 pixels, cada uma contendo um valor de 0 a 255 com seu valor em tons de cinza.

![image](https://user-images.githubusercontent.com/14276167/179138147-adadd93a-7dd7-4acf-ab1b-fa176a44af12.png)

É um subconjunto de um conjunto maior disponível no NIST. Os dígitos foram normalizados em tamanho e centralizados em uma imagem de tamanho fixo.

É um bom banco de dados para pessoas que desejam experimentar técnicas de aprendizado e métodos de reconhecimento de padrões em dados do mundo real, gastando esforços mínimos em pré-processamento e formatação.

Existem quatro arquivos disponíveis, que contêm separadamente treinar e testar, além de imagens e rótulos.

## Convolutional neural networks 
As redes neurais convolucionais se distinguem de outras redes neurais por terem desempenho superior com dados de imagem e áudio. Eles têm 3 camadas, à medida que as formas são revestidas pelos dados da imagem progridem da CNN, ela começa a reconhecer elementos ou maiores até mesmo identificar o objeto identificado ou objeto. A cada camada, a CNN aumenta em sua complexidade, identificando maiores porções da imagem.

convolutional layer

pool layer

Fully connected (FC) layer

#### convolutional layer
A camada convolucional é a parte central de uma CNN, é onde ocorre a maior parte da computação. Ela precisa de alguns componentes, que são dados de entrada, um filtro e um mapa de recursos. 

Considerando uma imagem colorida, composta por uma matriz de pixels em 3D. A entrada terá três dimensões — altura, largura e profundidade — que correspondem ao RGB em uma imagem. Também temos um detector de feição, também conhecido como kernel ou filtro, que irá percorrer os campos receptivos da imagem, verificando se a feição está presente. Esse processo é conhecido como convolução.

Embora possa variar em tamanho, o tamanho do filtro é normalmente uma matriz 3x3; isso também determina o tamanho do campo receptivo. O filtro é  aplicado a uma área da imagem em que um produto escalar é calculado entre os pixels de entrada e o filtro. Este produto escalar é então alimentado em uma matriz de saída. Depois, o filtro muda um passo, repetindo o processo até que o kernel tenha varrido toda a imagem. A saída final da série de produtos escalares da entrada e do filtro é conhecida como mapa de recursos, mapa de ativação ou recurso convoluído.

Na imagem abaixo vemos que , cada valor de saída no mapa de recursos não precisa se conectar a cada valor de pixel na imagem de entrada. Ele só precisa se conectar ao campo receptivo, onde o filtro está sendo aplicado.

![image](https://user-images.githubusercontent.com/14276167/179381754-89dfc357-6549-4854-9559-efc1f8070ef1.png)

Os pesos no detector de recursos permanecem fixos à medida que ele se move pela imagem, isso é conhecido como compartilhamento de parâmetros. Alguns parâmetros, como os valores de peso, se ajustam durante o treinamento através do processo de retropropagação e gradiente descendente. No entanto, existem três hiperparâmetros que afetam o tamanho do volume da saída que precisam ser definidos antes do início do treinamento da rede neural. 

Esses incluem:

1. O número de filtros -> afeta a profundidade da saída. Por exemplo, três filtros distintos produziriam três mapas de recursos diferentes, criando uma profundidade de três.

2. Stride -> é a distância, ou número de pixels, que o kernel se move sobre a matriz de entrada. Embora valores de passada de dois ou mais sejam raros, uma passada maior produz uma saída menor.

3. Zero padding ->  geralmente é usado quando os filtros não se ajustam à imagem de entrada. Isso define todos os elementos que estão fora da matriz de entrada para zero, produzindo uma saída maior ou de tamanho igual. 

Após cada operação de convolução, uma CNN aplica uma transformação de Unidade Linear Retificada (ReLU) ao mapa de características, para introduzir a não linearidade no modelo.

#### Pooling Layer
Conduz a redução de dimensionalidade, reduzindo o número de parâmetros na entrada. Varre um filtro em toda a entrada, esse filtro não possui pesose sim um kernel que aplica uma função de agregação aos valores dentro do campo receptivo e preenche a matriz de saída.

#### Fully-Connected Layer
Essa camada realiza a tarefa de classificação com base nas características extraídas das camadas anteriores e seus diferentes filtros. Enquanto as camadas convolucionais e de pooling tendem a usar funções ReLu, as camadas FC geralmente aproveitam uma função de ativação softmax para classificar as entradas adequadamente, produzindo uma probabilidade de 0 a 1.

Na camada Fully-Connected Layer, cada nó na camada de saída se conecta diretamente a um nó na camada anterior.

### Implementação do Convolutional neural networks para encontrar semelhanças de imagem
- Configurando a estrutura básica do código com transformações básicas
- Arquitetura do modelo de design com menos de 8.000 parâmetros

## Siamese Network
Nas Siamese Network existem dois conjuntos idênticos de camadas convolucionais que realmente compartilham filtros, ou seja eles têm duas entradas e têm apenas uma saída, então você tem duas imagens de entrada, por exemplo, e apenas uma pontuação de saída.

Nesse modelo pode receber duas entradas e dizer como elas são semelhantes, obviamente, as redes neurais convolucionais são realmente boas em identificar recursos, o que elas fazem primeiro é extrair os vetores de recursos de cada imagem e depois eles são recombinados em um único vetor, tomando a diferença absoluta entre os dois vetores de recursos, finalmente, o vetor de recursos único é colocado na função sigmoid que gera uma pontuação de semelhança e isso nos diz em alto nível o quão semelhantes essas duas entradas realmente são.

![image](https://user-images.githubusercontent.com/14276167/179136944-45658e8b-e13c-4278-b516-2971fb572414.png)

Exemplo para que possamos visualizar todo o seu conceito,

Digamos que queremos inserir duas faces em nosso modelo e ver como elas são semelhantes ou se pertencem à mesma pessoa nas duas imagens de entrada. Se forem semelhantes, elas teriam vetores de recursos semelhantes, isso significara uma diferença absoluta mais baixa e, portanto, uma pontuação de semelhança mais alta.

### implementação da Siamese Network para encontrar semelhanças de imagem
#### Arquitetura utlizada
![image](https://user-images.githubusercontent.com/14276167/179137184-b2790515-bdc4-43fe-9644-351ec5f7be06.png)

#### Resultados
![image](https://user-images.githubusercontent.com/14276167/179141161-85485c9a-8e76-4f45-94b8-c5c63deed31b.png)

