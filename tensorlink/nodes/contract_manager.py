from typing import List, Tuple, Optional, Dict, Any
from web3.exceptions import ContractLogicError
from dotenv import get_key
from eth_abi import encode
import logging
import hashlib
import threading
import time


class ContractManager:
    """
    Manages blockchain contract interactions for validator proposals and job management.

    This class handles the creation, submission, and execution of proposals for
    validator removal and job completion on the blockchain.
    """

    def __init__(self, node, multi_sig_contract, chain, public_key: str):
        """
        Initialize the Contract Manager.

        Args:
            node: Parent node instance that contains DHT query methods
            multi_sig_contract: Web3 contract instance for multi-signature operations
            chain: Web3 chain connection
            public_key: Public key of the node
        """
        self.node = node
        self.multi_sig_contract = multi_sig_contract
        self.chain = chain
        self.public_key = public_key

        # State tracking
        self.validators_to_clear: List[str] = self.node.validators_to_clear
        self.jobs_to_complete: List[str] = self.node.jobs_to_complete
        self.current_proposal: Optional[int] = (
            self.multi_sig_contract.functions.nextProposalId.call()
        )
        self.terminate_flag = node.terminate_flag

        self.proposals = {}

    def proposal_validator(self):
        """Listen for new proposals created on SmartnodesMultiSig and validate them."""
        current_proposal_num = 1

        # Initialize last_execution_block from existing proposals
        while not self.terminate_flag.is_set():
            try:
                # Check if a new round of proposals has started
                current_proposal_id = (
                    self.multi_sig_contract.functions.nextProposalId().call()
                )
                time.sleep(1)

                # Update variables for new round of proposals
                if current_proposal_id != self.current_proposal:
                    self.current_proposal = current_proposal_id
                    self.proposals = {}
                    current_proposal_num = 1

                # Get ID for next proposal in round
                try:
                    # If proposal exists, get its hash
                    proposal_hash = (
                        self.multi_sig_contract.functions.currentProposals(
                            current_proposal_num
                            - 1  # index for querying next proposal candidate
                        )
                        .call()
                        .hex()
                    )
                    time.sleep(1)

                except ContractLogicError:
                    # Proposal has not been published yet, keep waiting
                    time.sleep(60)
                    continue

                # Validator proposal
                t = threading.Thread(
                    target=self.validate_proposal,
                    args=(proposal_hash, current_proposal_num),
                    name=f"proposal_validator_{current_proposal_num}",
                    daemon=True,
                )
                self.proposals[proposal_hash] = t
                t.start()
                current_proposal_num += 1
                time.sleep(1)

                if current_proposal_num > 1:
                    time.sleep(3000)

            except Exception as e:
                self.node.debug_print(
                    f"Validator -> Error while fetching created proposals: {e}",
                    colour="bright_red",
                    level=logging.ERROR,
                )

            time.sleep(3)

    def validate_proposal(self, proposal_hash, proposal_num):
        # TODO if we are the proposal creator (ie a selected validator), automatically cast a vote.
        #  We should also use our proposal data to quickly verify matching data
        self.node.debug_print(
            f"Validator -> Validation started for proposal: {proposal_hash}",
            colour="bright_blue",
            level=logging.INFO,
        )

        # Query network for the detailed proposal info
        proposal_data = self.node.query_dht(proposal_hash)

        # If no proposal is found
        if proposal_data is None:
            # Perhaps some logic to directly query the node who published the proposal to ensure it is not found TODO
            self.node.debug_print(
                f"ContractManager -> Validating proposal {proposal_hash} not found in DHT!"
            )
            return

        proposal_data_hash = self._hash_proposal_data(proposal_data).hex()

        if proposal_data_hash != proposal_hash:
            # Proposal hash must match smart contract listed proposal
            self.node.debug_print("Validator -> Invalid proposal hash!", colour="red")
            return

        self._approve_transaction(proposal_num, proposal_hash)

        # for function_type, call_data in proposal_data:
        #     if self.proposal_flag.is_set():
        #         self.debug_print(f"Validator -> Validation interrupted for proposal {proposal_data_hash}!")
        #         return
        #
        #     # Deactivate validator call: attempt connection to determine if validator should be deactivated
        #     if function_type == 0:
        #         node_address = self.chain.to_checksum_address(
        #             decode(["address"], call_data)[0]
        #         )
        #         node_hash = self.contract.functions.getValidatorBytes(node_address).call().hex()
        #         node_info = self.query_dht(node_hash)
        #
        #         if node_info:
        #             node_host, node_port = node_info["host"], node_info["port"]
        #             connected = self.connect_node(node_info["id"], node_host, node_port)
        #
        #             # Verify the roles is online and in the network
        #             if connected:
        #                 flag = True
        #                 reason = f"Validator still online."
        #                 connection = self.nodes[node_info]
        #                 self.close_connection_socket(connection, "Heart beat complete.")
        #
        #         else:
        #             flag = True
        #             reason = f"Node not found: {node_hash}"
        #
        #     # Create job, ensure job is requested on p2p network
        #     elif function_type == 1:
        #         # TODO ensure the job type is for the tensorlink sub-network
        #         user_hash, job_hash, capacities = decode(
        #             ["bytes32", "bytes32", "uint256[]"], call_data
        #         )
        #
        #         user_hash = user_hash.hex()
        #         job_hash = job_hash.hex()
        #
        #         validated = self.validate_job(job_hash, user_id=user_hash, capacities=capacities)
        #
        #         if not validated:
        #             flag = True
        #             reason = f"Invalid job state! {job_hash}"
        #
        #     # Complete job, determine status on p2p network
        #     elif function_type == 2:
        #         job_hash, worker_addresses = decode(
        #             ["bytes32", "address[]"], call_data
        #         )
        #
        #         job_hash = job_hash.hex()
        #
        #         validated = self.validate_job(job_hash, active=False)
        #
        #         if not validated:
        #             flag = True
        #             reason = f"Invalid job state! {job_hash}"
        #
        #     else:
        #         flag = True
        #         reason = "Invalid function type."
        #
        #     if flag is True:
        #         self.debug_print(f"Validator -> Job validation failed: {reason}", colour="bright_red",
        #         level=logging.WARNING)
        #         break

    def _approve_transaction(self, proposal_num: int, proposal_hash: str):
        try:
            # Determine if proposal can be submitted
            tx = self.multi_sig_contract.functions.approveTransaction(
                proposal_num
            ).build_transaction(
                {
                    "from": self.public_key,
                    "nonce": self.chain.eth.get_transaction_count(self.public_key),
                    "gas": 6721975,
                    "gasPrice": self.chain.eth.gas_price,
                }
            )
            signed_tx = self.chain.eth.account.sign_transaction(
                tx, get_key(".tensorlink.env", "PRIVATE_KEY")
            )
            tx_hash = self.chain.eth.send_raw_transaction(signed_tx.raw_transaction)
            self.node.debug_print(
                f"Validator -> Proposal {proposal_num}: {proposal_hash} approved! ({tx_hash.hex()})",
                colour="green",
                level=logging.INFO,
            )

        except Exception as e:
            if "Validator has already voted!" in str(e):
                self.node.debug_print(
                    f"Validator -> Have already voted on proposal {proposal_num}, continuing...",
                    colour="green",
                    level=logging.DEBUG,
                )
                pass
            else:
                raise e

    def add_validator_to_clear(self, validator_id: str) -> None:
        """Add a validator to the list of validators to be cleared."""
        if validator_id not in self.validators_to_clear:
            self.validators_to_clear.append(validator_id)

    def add_job_to_complete(self, job_data: dict) -> None:
        """Add a job to the list of jobs to be completed."""
        # Update job to contract if a certain threshold of work was done
        if (
            job_data["timestamp"] - job_data["last_seen"] > 180
            and job_data["gigabyte_hours"] > 5e8
        ) and job_data["id"] not in self.jobs_to_complete:
            self.jobs_to_complete.append(job_data["id"])

    def verify_and_remove_validators(self) -> List[str]:
        """
        Verify validator status and create list of offline validators to remove.

        Returns:
            List[str]: List of validator addresses to be removed
        """
        validators_to_remove = []

        for validator in self.node.validators_to_clear:
            node_info = self.node.query_dht(validator)
            if not node_info:
                continue

            if not self._is_validator_online(node_info):
                node_address = self._get_validator_address(validator)
                if node_address:
                    validators_to_remove.append(node_address)

        return validators_to_remove

    def process_jobs(self) -> Tuple[List[bytes], List[int], List[str]]:
        """
        Process jobs to be completed and collect necessary information.

        Returns:
            Tuple containing:
            - List[bytes]: Job hashes
            - List[int]: Job capacities
            - List[str]: Worker addresses
        """
        job_hashes = []
        job_capacities = []
        job_workers = []

        for job_id in self.jobs_to_complete:
            job = self.node.query_dht(job_id)
            if not job:
                continue

            job_data = self._process_single_job(job, job_id)
            if job_data:
                job_hash, capacity, workers = job_data
                job_hashes.append(job_hash)
                job_capacities.append(capacity)
                job_workers.extend(workers)

        return job_hashes, job_capacities, job_workers

    def proposal_creator(self):
        while not self.terminate_flag.is_set():
            try:
                # Fetch state from the contract
                (
                    next_proposal_id,
                    round_validators,
                ) = self.multi_sig_contract.functions.getState().call()
                time.sleep(1)
                is_expired = self.multi_sig_contract.functions.isRoundExpired().call()
                time.sleep(1)

                if self.public_key in round_validators or is_expired:
                    # Wait a bit before creating the proposal
                    self.create_and_submit_proposal()
                    time.sleep(3000)

            except Exception as e:
                self.node.debug_print(
                    f"Validator -> Error processing new entries: {e}",
                    colour="bright_red",
                    level=logging.ERROR,
                )

            time.sleep(60)

    def create_and_submit_proposal(self) -> None:
        """
        Main function to create and execute a proposal.
        Coordinates the process of creating, submitting, and executing a proposal
        for validator removal and job completion.
        """
        self.node.debug_print(
            "ContractManager -> Creating proposal...",
            colour="bright_blue",
            level=logging.INFO,
        )

        while True:
            # Verify proposal can be submitted
            try:
                self.multi_sig_contract.functions.createProposal(
                    encode(["uint256"], [12345])
                ).call({"from": self.public_key})

            except Exception:
                # If the proposal couldnt be created, we must wait
                time.sleep(120)
                pass

            # Request all workers connected to the network
            self.node.get_workers()

            # Process validators and jobs
            validators_to_remove = self.verify_and_remove_validators()
            job_hashes, job_capacities, job_workers = self.process_jobs()

            total_capacities = [
                int(
                    sum(
                        worker["total_gpu_memory"]
                        for worker in self.node.all_workers.values()
                    )
                )
            ]
            total_workers = [len(self.node.all_workers)]

            # Create and store proposal
            proposal = {
                "validators": validators_to_remove,
                "jobs": job_hashes,
                "job_capacities": job_capacities,
                "workers": job_workers,
                "total_capacity": total_capacities,
                "total_workers": total_workers,
            }
            proposal_hash = self._hash_proposal_data(proposal)
            self.node.store_value(proposal_hash.hex(), proposal)
            self.proposals[proposal_hash.hex()] = proposal

            # Submit proposal
            code = self._submit_proposal(proposal_hash)

            if code == 0:
                break  # Exit loop if already submitted or submission is successful
            elif code == 1:
                return  # Exit method (error)
            elif code == 2:
                time.sleep(30)
                continue  # Restart and continue to build proposal data

        # Wait for next block before monitoring
        self._wait_for_next_block()

        # Monitor and execute proposal
        self._monitor_and_execute_proposal(
            validators_to_remove,
            job_hashes,
            job_capacities,
            job_workers,
            total_capacities,
            total_workers,
        )

    def _is_validator_online(self, node_info: Dict[str, Any]) -> bool:
        """Check if a validator is online and connected to the network."""
        node_host, node_port = node_info["host"], node_info["port"]
        return self.node.connect_node(node_info["id"], node_host, node_port)

    def _get_validator_address(self, validator: str) -> Optional[str]:
        """Get the blockchain address for a validator."""
        return self.node.contract.functions.validatorAddressByHash(validator).call()

    def _process_single_job(
        self, job: Dict[str, Any], job_id: str
    ) -> Optional[Tuple[bytes, int, List[str]]]:
        """Process a single job and return its data."""
        job_hash = bytes.fromhex(job_id)
        capacities = job["capacity"]
        workers = []

        for module_info in job["distribution"].values():
            for worker_id in module_info["workers"]:
                worker_address = self._get_worker_address(worker_id)
                if worker_address:
                    workers.append(worker_address)
                else:
                    workers.append(self.public_key)

        return job_hash, capacities, workers

    def _get_worker_address(self, worker_id: str) -> Optional[str]:
        """Get the blockchain address for a worker."""
        worker_info = self.node.query_dht(worker_id)
        if not worker_info:
            return None

        if not self._is_validator_online(worker_info):
            return None

        worker_node = self.node.nodes[worker_id]
        return self.node.query_node(hashlib.sha256(b"ADDRESS").hexdigest(), worker_node)

    def _hash_proposal_data(self, proposal_data: dict):
        (validators_to_remove, job_hashes, job_capacities, job_workers, _, _) = (
            proposal_data.values()
        )

        validators_to_remove = [
            self.chain.to_checksum_address(validator)
            for validator in validators_to_remove
        ]
        workers = [self.chain.to_checksum_address(worker) for worker in job_workers]
        encoded_data = encode(
            ["address[]", "bytes32[]", "uint256[]", "address[]"],
            [validators_to_remove, job_hashes, job_capacities, workers],
        )

        return self.chain.keccak(encoded_data)

    def _submit_proposal(self, proposal_hash: bytes) -> int:
        """Submit the proposal to the blockchain."""
        while not self.terminate_flag.is_set():
            try:
                # Verify proposal can be submitted
                self.multi_sig_contract.functions.createProposal(proposal_hash).call(
                    {"from": self.public_key}
                )

                # Build and submit transaction
                tx = self._build_proposal_transaction(proposal_hash)
                tx_hash = self._submit_transaction(tx)

                self.node.debug_print(
                    f"ContractManager -> Proposal ({proposal_hash.hex()}) submitted! ({tx_hash.hex()})",
                    colour="green",
                    level=logging.INFO,
                )
                return 0

            except Exception as e:
                if "Validator has already submitted a proposal this round" in str(e):
                    self.node.debug_print(
                        "ContractManager -> Validator has already submitted a proposal this round!",
                        colour="bright_red",
                        level=logging.INFO,
                    )
                    return 0

                elif "updateTime - 2min" in str(e):
                    self.node.debug_print(
                        "ContractManager -> Not enough time since last proposal! Sleeping...",
                        colour="green",
                        level=logging.DEBUG,
                    )
                    time.sleep(60)
                    return 2
                else:
                    self.node.debug_print(
                        f"ContractManager -> Error creating proposal: {str(e)}",
                        colour="bright_red",
                        level=logging.INFO,
                    )
                    return 1

    def _build_proposal_transaction(self, proposal_hash: bytes) -> Dict[str, Any]:
        """Build the proposal transaction."""
        return self.multi_sig_contract.functions.createProposal(
            proposal_hash
        ).build_transaction(
            {
                "from": self.public_key,
                "nonce": self.chain.eth.get_transaction_count(self.public_key),
                "gas": 6_721_975,
                "gasPrice": self.chain.eth.gas_price,
            }
        )

    def _submit_transaction(self, tx: Dict[str, Any]) -> str:
        """Submit a transaction to the blockchain."""
        signed_tx = self.chain.eth.account.sign_transaction(
            tx, get_key(".tensorlink.env", "PRIVATE_KEY")
        )
        return self.chain.eth.send_raw_transaction(signed_tx.raw_transaction)

    def _wait_for_next_block(self) -> None:
        """Wait for the next blockchain block."""
        current_block = self.chain.eth.block_number
        while True:
            new_block = self.chain.eth.block_number
            if new_block > current_block:
                break
            time.sleep(5)

    def _monitor_and_execute_proposal(
        self,
        validators_to_remove: List[str],
        job_hashes: List[bytes],
        job_capacities: List[int],
        job_workers: List[str],
        total_capacities: List[int],
        total_workers: List[int],
    ) -> None:
        """Monitor proposal status and execute when ready."""
        while not self.terminate_flag.is_set():
            if not self._is_proposal_valid():
                return

            if self._is_proposal_ready():
                self._execute_proposal(
                    validators_to_remove,
                    job_hashes,
                    job_capacities,
                    job_workers,
                    total_capacities,
                    total_workers,
                )
                return

            time.sleep(15)

    def _is_proposal_valid(self) -> bool:
        """Check if the current proposal is still valid."""
        proposal_id = self.multi_sig_contract.functions.nextProposalId.call()
        time.sleep(1)
        return self.current_proposal == proposal_id

    def _is_proposal_ready(self) -> bool:
        """Check if the proposal is ready for execution."""
        proposal_number = self.multi_sig_contract.functions.hasSubmittedProposal(
            self.public_key
        ).call()
        time.sleep(1)
        return self.multi_sig_contract.functions.isProposalReady(proposal_number).call()

    def _execute_proposal(
        self,
        validators_to_remove: List[str],
        job_hashes: List[bytes],
        job_capacities: List[int],
        job_workers: List[str],
        total_capacities: List[int],
        total_workers: List[int],
    ) -> bool:
        """Execute the proposal once it's ready."""
        try:
            self.multi_sig_contract.functions.executeProposal(
                validators_to_remove,
                job_hashes,
                job_capacities,
                job_workers,
                total_capacities,
                total_workers,
            ).call({"from": self.public_key})

            execute_tx = self._build_execution_transaction(
                validators_to_remove,
                job_hashes,
                job_capacities,
                job_workers,
                total_capacities,
                total_workers,
            )
            execute_tx_hash = self._submit_transaction(execute_tx)

            self.node.debug_print(
                f"ContractManager -> Proposal executed! ({execute_tx_hash.hex()})",
                colour="green",
                level=logging.INFO,
            )

            self._clear_completed_items()
            return True

        except Exception as e:
            self._handle_execution_error(e)
            return False

    def _build_execution_transaction(
        self,
        validators_to_remove: List[str],
        job_hashes: List[bytes],
        job_capacities: List[int],
        job_workers: List[str],
        total_capacities: List[int],
        total_workers: List[int],
    ) -> Dict[str, Any]:
        """Build the execution transaction"""
        return self.node.multi_sig_contract.functions.executeProposal(
            validators_to_remove,
            job_hashes,
            job_capacities,
            job_workers,
            total_capacities,
            total_workers,
        ).build_transaction(
            {
                "from": self.public_key,
                "nonce": self.chain.eth.get_transaction_count(self.public_key),
                "gas": 6_721_975,
                "gasPrice": self.chain.eth.gas_price,
            }
        )

    def _handle_execution_error(self, error: Exception) -> None:
        """Handle errors during proposal execution."""
        if "Not enough proposal votes!" in str(error):
            self.node.debug_print(
                "ContractManager -> Not enough proposal votes, sleeping...",
                colour="green",
                level=logging.DEBUG,
            )
        else:
            self.node.debug_print(
                f"ContractManager -> Error executing proposal: {error}",
                colour="bright_red",
                level=logging.ERROR,
            )

    def _clear_completed_items(self) -> None:
        """Clear lists of completed validators and jobs."""
        self.validators_to_clear = []
        self.jobs_to_complete = []
